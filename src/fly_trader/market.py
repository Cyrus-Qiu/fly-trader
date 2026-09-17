from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import asyncio
import json
import os
from collections.abc import AsyncIterator

import pandas as pd


def _is_authentication_failure(error: Exception) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in (
        "auth failed", "authentication failed", "unauthorized",
        "permission denied", "status 401", "status 403",
        "http 401", "http 403",
    ))


@dataclass(frozen=True)
class MarketSnapshot:
    symbol: str
    market_time_ms: int
    fetched_at: str
    pre_close: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    halted: int
    feed: str = "unknown"
    event_type: str = "snapshot"
    bid: float | None = None
    ask: float | None = None
    spread_bps: float | None = None

    def public_dict(self) -> dict:
        return asdict(self)


class TimestampDeduplicator:
    """Accept each symbol/provider timestamp at most once, including out-of-order data."""

    def __init__(self) -> None:
        self._seen: set[tuple[str, str, int]] = set()
        self._latest: dict[tuple[str, str], int] = {}

    def accept(self, snapshot: MarketSnapshot) -> bool:
        source = (snapshot.symbol, snapshot.feed)
        key = (*source, snapshot.market_time_ms)
        if key in self._seen:
            return False
        latest = self._latest.get(source)
        if latest is not None and snapshot.market_time_ms < latest:
            return False
        self._seen.add(key)
        self._latest[source] = snapshot.market_time_ms
        return True


class TigerDelayedQuoteSource:
    """Read-only adapter. It constructs QuoteClient only; no trade client exists here."""

    def __init__(self, config_path: Path):
        from tigeropen.quote.quote_client import QuoteClient
        from tigeropen.tiger_open_config import TigerOpenClientConfig

        config = TigerOpenClientConfig(props_path=str(config_path))
        self._client = QuoteClient(config)

    def fetch(self, symbols: list[str]) -> list[MarketSnapshot]:
        frame = self._client.get_stock_delay_briefs(symbols)
        return snapshots_from_frame(frame)


class IEXTradeAccumulator:
    """Convert Alpaca IEX trades into monotonically timestamped OHLCV snapshots."""

    def __init__(self) -> None:
        self._state: dict[str, dict[str, float]] = {}

    def ingest(self, message: dict) -> MarketSnapshot | None:
        if message.get("T") != "t":
            return None
        symbol = str(message["S"]).upper()
        price = float(message["p"])
        size = float(message.get("s", 0))
        timestamp = int(pd.Timestamp(message["t"]).value // 1_000_000)
        state = self._state.setdefault(symbol, {
            "reference": price, "open": price, "high": price, "low": price,
            "close": price, "volume": 0.0,
        })
        state["high"] = max(state["high"], price)
        state["low"] = min(state["low"], price)
        state["close"] = price
        state["volume"] += size
        return MarketSnapshot(
            symbol=symbol,
            market_time_ms=timestamp,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            # Until a session bootstrap endpoint is added, pre_close is the
            # first IEX trade observed after this process connects.
            pre_close=state["reference"],
            open=state["open"], high=state["high"], low=state["low"],
            close=state["close"], volume=state["volume"], halted=0,
            feed="iex", event_type="trade",
        )


class OvernightQuoteAccumulator:
    """Convert free real-time indicative overnight quotes into midpoint OHLC."""

    def __init__(self, max_spread_bps: float = 100.0, max_jump_fraction: float = 0.02) -> None:
        self._state: dict[str, dict[str, float]] = {}
        self.max_spread_bps = max_spread_bps
        self.max_jump_fraction = max_jump_fraction

    def ingest(self, message: dict) -> MarketSnapshot | None:
        if message.get("T") != "q":
            return None
        bid, ask = float(message.get("bp", 0)), float(message.get("ap", 0))
        if bid <= 0 or ask <= 0 or ask < bid:
            return None
        symbol = str(message["S"]).upper()
        midpoint = (bid + ask) / 2.0
        spread_bps = (ask - bid) / midpoint * 10_000.0
        if spread_bps > self.max_spread_bps:
            return None
        timestamp = int(pd.Timestamp(message["t"]).value // 1_000_000)
        state = self._state.setdefault(symbol, {
            "reference": midpoint, "open": midpoint, "high": midpoint,
            "low": midpoint, "close": midpoint,
        })
        if state["close"] > 0 and abs(midpoint / state["close"] - 1.0) > self.max_jump_fraction:
            return None
        state["high"] = max(state["high"], midpoint)
        state["low"] = min(state["low"], midpoint)
        state["close"] = midpoint
        return MarketSnapshot(
            symbol=symbol, market_time_ms=timestamp,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            pre_close=state["reference"], open=state["open"],
            high=state["high"], low=state["low"], close=state["close"],
            volume=0.0, halted=0, feed="overnight",
            event_type="indicative_quote",
            bid=bid, ask=ask, spread_bps=spread_bps,
        )


class AlpacaMarketSource:
    """Read-only Alpaca market-data stream; contains no brokerage client."""

    urls = {
        "iex": "wss://stream.data.alpaca.markets/v2/iex",
        "overnight": "wss://stream.data.alpaca.markets/v1beta1/overnight",
    }

    def __init__(self, symbols: list[str], feed: str = "iex",
                 key: str | None = None, secret: str | None = None):
        if feed not in self.urls:
            raise ValueError(f"unsupported Alpaca feed: {feed}")
        self.feed = feed
        self.url = self.urls[feed]
        self.symbols = [symbol.upper() for symbol in symbols]
        self.key = key or os.getenv("APCA_API_KEY_ID")
        self.secret = secret or os.getenv("APCA_API_SECRET_KEY")
        if not self.key or not self.secret:
            raise RuntimeError("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in the environment")
        self.accumulator = IEXTradeAccumulator() if feed == "iex" else OvernightQuoteAccumulator()
        self.channel = "trades" if feed == "iex" else "quotes"

    async def stream(self) -> AsyncIterator[MarketSnapshot]:
        import websockets

        delay = 1.0
        while True:
            try:
                async with websockets.connect(self.url, ping_interval=20, ping_timeout=20) as socket:
                    print(f"已连接 Alpaca 行情：{self.url}")
                    await socket.send(json.dumps({"action": "auth", "key": self.key, "secret": self.secret}))
                    authenticated = False
                    async for raw in socket:
                        messages = json.loads(raw)
                        for message in messages:
                            if message.get("T") == "success" and message.get("msg") == "authenticated":
                                authenticated = True
                                print("Alpaca 鉴权成功")
                                await socket.send(json.dumps({"action": "subscribe", self.channel: self.symbols}))
                                label = "IEX 实时成交" if self.feed == "iex" else "隔夜实时指示性报价"
                                print(f"已订阅 {label}：{', '.join(self.symbols)}")
                                delay = 1.0
                                continue
                            if message.get("T") == "error":
                                code = int(message.get("code", 0))
                                error = RuntimeError(f"Alpaca stream error {code}: {message.get('msg')}")
                                if 400 <= code < 500:
                                    raise PermissionError(str(error))
                                raise error
                            snapshot = self.accumulator.ingest(message)
                            if snapshot is not None:
                                yield snapshot
                    if not authenticated:
                        raise RuntimeError("Alpaca stream closed before authentication")
            except (asyncio.CancelledError, PermissionError):
                raise
            except Exception as error:
                print(f"{self.feed} 连接中断：{type(error).__name__}；{delay:.0f} 秒后重试")
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)


def _longbridge_snapshot(symbol: str, quote: object, feed: str,
                         event_type: str) -> MarketSnapshot:
    """Convert Longbridge SecurityQuote/PushQuote without logging credentials."""
    timestamp = getattr(quote, "timestamp")
    if isinstance(timestamp, datetime):
        market_time_ms = int(timestamp.timestamp() * 1000)
    else:
        market_time_ms = int(pd.Timestamp(timestamp).value // 1_000_000)
    status = str(getattr(quote, "trade_status", "")).upper()
    return MarketSnapshot(
        symbol=symbol.upper(), market_time_ms=market_time_ms,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        pre_close=float(getattr(quote, "prev_close", 0) or 0),
        open=float(getattr(quote, "open", 0) or 0),
        high=float(getattr(quote, "high", 0) or 0),
        low=float(getattr(quote, "low", 0) or 0),
        close=float(getattr(quote, "last_done", 0) or 0),
        volume=float(getattr(quote, "volume", 0) or 0),
        halted=int("HALT" in status or "SUSPEND" in status),
        feed=feed, event_type=event_type,
    )


class LongbridgeHKMarketSource:
    """Read-only HK quote adapter with realtime capability detection and BMP fallback."""

    def __init__(self, symbols: list[str], poll_seconds: float = 5.0):
        self.symbols = [self._normalize_symbol(symbol) for symbol in symbols]
        self.poll_seconds = max(1.0, poll_seconds)
        required = ("LONGBRIDGE_APP_KEY", "LONGBRIDGE_APP_SECRET",
                    "LONGBRIDGE_ACCESS_TOKEN")
        missing = [name for name in required if not os.getenv(name)]
        if missing:
            raise RuntimeError("请在环境变量中设置：" + ", ".join(missing))

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        value = symbol.strip().upper()
        ticker = value[:-3] if value.endswith(".HK") else value
        return f"{ticker.lstrip('0') or '0'}.HK"

    @staticmethod
    def _pull_feed(quote: object) -> str:
        timestamp = getattr(quote, "timestamp")
        quote_time = timestamp if isinstance(timestamp, datetime) else pd.Timestamp(timestamp).to_pydatetime()
        if quote_time.tzinfo is None:
            quote_time = quote_time.replace(tzinfo=timezone.utc)
        lag = (datetime.now(timezone.utc) - quote_time.astimezone(timezone.utc)).total_seconds()
        return "longbridge-hk-bmp" if lag > 120 else "longbridge-hk-poll"

    async def stream(self) -> AsyncIterator[MarketSnapshot]:
        delay = 1.0
        while True:
            try:
                async for snapshot in self._stream_once():
                    delay = 1.0
                    yield snapshot
                raise RuntimeError("长桥行情连接意外结束")
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if _is_authentication_failure(error):
                    raise PermissionError(f"长桥鉴权或权限失败：{error}") from error
                print(f"长桥行情连接中断：{type(error).__name__}；{delay:.0f} 秒后重连")
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)

    async def _stream_once(self) -> AsyncIterator[MarketSnapshot]:
        from longbridge.openapi import AsyncQuoteContext, Config, SubType

        # Explicit construction prevents credentials from being read from a repository .env file.
        config = Config.from_apikey(
            os.environ["LONGBRIDGE_APP_KEY"],
            os.environ["LONGBRIDGE_APP_SECRET"],
            os.environ["LONGBRIDGE_ACCESS_TOKEN"],
            enable_print_quote_packages=False,
            enable_papertrading=True,
        )
        context = AsyncQuoteContext.create(config)
        print(f"已连接长桥港股行情；正在探测实时推送能力：{', '.join(self.symbols)}")
        initial = await context.quote(self.symbols)
        for quote in initial:
            yield _longbridge_snapshot(quote.symbol, quote, self._pull_feed(quote), "snapshot")

        queue: asyncio.Queue[MarketSnapshot] = asyncio.Queue(maxsize=4096)
        loop = asyncio.get_running_loop()

        def enqueue(snapshot: MarketSnapshot) -> None:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(snapshot)

        def on_quote(symbol: str, event: object) -> None:
            snapshot = _longbridge_snapshot(
                symbol, event, "longbridge-hk-realtime", "quote")
            loop.call_soon_threadsafe(enqueue, snapshot)

        context.set_on_quote(on_quote)
        try:
            await context.subscribe(self.symbols, [SubType.Quote])
            print(f"已订阅长桥港股报价；若 {self.poll_seconds:g} 秒内无推送，将主动拉取检查")
        except Exception as error:
            print(f"港股实时订阅不可用（{type(error).__name__}），自动改用定时拉取")
        while True:
            try:
                yield await asyncio.wait_for(queue.get(), timeout=self.poll_seconds)
            except asyncio.TimeoutError:
                for quote in await context.quote(self.symbols):
                    feed = self._pull_feed(quote)
                    if feed == "longbridge-hk-bmp":
                        print(f"权限探测｜{quote.symbol} 返回延迟时间戳，按 BMP 行情处理")
                    yield _longbridge_snapshot(quote.symbol, quote, feed, "snapshot")


def snapshots_from_frame(frame: pd.DataFrame) -> list[MarketSnapshot]:
    fetched_at = datetime.now(timezone.utc).isoformat()
    result: list[MarketSnapshot] = []
    for row in frame.to_dict(orient="records"):
        result.append(MarketSnapshot(
            symbol=str(row["symbol"]).upper(),
            market_time_ms=int(row["time"]),
            fetched_at=fetched_at,
            pre_close=float(row.get("pre_close") or 0),
            open=float(row.get("open") or 0),
            high=float(row.get("high") or 0),
            low=float(row.get("low") or 0),
            close=float(row.get("close") or 0),
            volume=float(row.get("volume") or 0),
            halted=int(row.get("halted") or 0),
            feed="tiger-delayed", event_type="snapshot",
        ))
    return result
