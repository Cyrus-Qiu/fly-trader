from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import asyncio
import os
from collections.abc import AsyncIterator

import pandas as pd


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
    trade_session: str = "regular"

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


class LongbridgeMarketSource:
    """One provider for HK regular hours and all four US sessions."""
    def __init__(self, symbols, poll_seconds=2.0, market="HK"):
        from .calendar import TradingCalendar
        self.market = market
        self.symbols = [self._normalize_symbol(s) for s in symbols]
        self.api_symbols = [s if market == "HK" else f"{s}.US" for s in self.symbols]
        self.poll_seconds = max(1.0, poll_seconds)
        self.calendar = TradingCalendar()
        self.context = None
        self.endpoint = None
        self.status = "connecting"
        self.status_reason = "正在连接长桥行情"
        self.references = {}
        required = ("LONGBRIDGE_APP_KEY", "LONGBRIDGE_APP_SECRET", "LONGBRIDGE_ACCESS_TOKEN")
        missing = [key for key in required if not os.getenv(key)]
        if missing:
            raise RuntimeError("请设置长桥行情凭据：" + ", ".join(missing))

    @staticmethod
    def _normalize_symbol(symbol):
        value = symbol.strip().upper()
        ticker = value.removesuffix(".HK")
        return f"{ticker.lstrip('0') or '0'}.HK"

    async def initialize(self):
        from longbridge.openapi import AsyncQuoteContext, Config
        if self.context is not None:
            return
        custom = os.getenv("LONGBRIDGE_HTTP_URL")
        endpoints = [custom] if custom else ["https://openapi.longbridge.cn", "https://openapi.longbridge.com"]
        for endpoint in endpoints:
            try:
                quote_ws = os.getenv("LONGBRIDGE_QUOTE_WS_URL") or (
                    "wss://openapi-quote.longbridge.cn" if endpoint.endswith(".cn") else "wss://openapi-quote.longbridge.com")
                config = Config.from_apikey(
                    os.environ["LONGBRIDGE_APP_KEY"], os.environ["LONGBRIDGE_APP_SECRET"],
                    os.environ["LONGBRIDGE_ACCESS_TOKEN"], http_url=endpoint, quote_ws_url=quote_ws,
                    enable_overnight=True, enable_print_quote_packages=False,
                    enable_papertrading=False)
                context = AsyncQuoteContext.create(config)
                await asyncio.wait_for(context.quote(self.api_symbols), 10)
                self.context, self.endpoint = context, endpoint
                self.status, self.status_reason = "connected", "长桥行情已连接"
                return
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self.status_reason = f"{type(error).__name__}：长桥接入点连接或行情权限不可用"
        self.status = "retrying"
        raise ConnectionError(self.status_reason)

    async def lot_sizes(self):
        await self.initialize()
        info = await asyncio.wait_for(self.context.static_info(self.api_symbols), 10)
        return {self._normalize_symbol(item.symbol): int(item.lot_size)
                for item in info if item.lot_size > 0}

    def snapshots(self, quote, now_ms=None):
        import time
        from dataclasses import replace
        now_ms = int(time.time() * 1000) if now_ms is None else now_ms
        symbol = self._normalize_symbol(quote.symbol)
        if self.market == "HK" and self.calendar.session_reason(symbol, now_ms) is not None:
            self.status_reason = f"{symbol} 当前不在港股连续交易时段"
            return []
        session = self.calendar.us_session(now_ms) if self.market == "US" else "regular"
        if session == "closed":
            return []
        selected = (getattr(quote, {"pre": "pre_market_quote", "post": "post_market_quote",
                                 "overnight": "overnight_quote"}[session], None)
                    if session != "regular" else quote)
        if selected is None:
            self.status_reason = f"{symbol} 缺少当前 {session} 时段报价"
            return []
        feed = f"longbridge-{self.market.lower()}-{session}"
        snapshot = _longbridge_snapshot(symbol, selected, feed, "snapshot")
        # SDK datetimes without tzinfo represent machine-local time: timestamp()
        # already converts these to UTC. Do not relabel a local naive time as UTC.
        if snapshot.close <= 0:
            return []
        if now_ms - snapshot.market_time_ms > 120_000:
            snapshot = replace(snapshot, feed=feed + "-stale")
        status = str(getattr(quote, "trade_status", "")).upper()
        reference = snapshot.pre_close or snapshot.close
        self.references[(symbol, session)] = reference
        snapshot = replace(snapshot, trade_session=session,
                           pre_close=reference, open=snapshot.open or reference,
                           high=snapshot.high or snapshot.close, low=snapshot.low or snapshot.close,
                           halted=int("HALT" in status or "SUSPEND" in status))
        return [snapshot]

    def push_snapshot(self, symbol, event):
        import time
        from dataclasses import replace
        symbol = self._normalize_symbol(symbol)
        session = str(getattr(event, "trade_session", "Intraday")).split(".")[-1].lower()
        session = {"intraday": "regular", "pre": "pre", "post": "post", "overnight": "overnight"}.get(session)
        now_ms = int(time.time() * 1000)
        if self.market == "HK" and self.calendar.session_reason(symbol, now_ms) is not None:
            self.status_reason = f"{symbol} 当前不在港股连续交易时段"
            return None
        expected = self.calendar.us_session(now_ms) if self.market == "US" else "regular"
        if session != expected or expected == "closed":
            return None
        snapshot = _longbridge_snapshot(symbol, event, f"longbridge-{self.market.lower()}-{session}", "quote")
        if snapshot.close <= 0:
            return None
        reference = self.references.setdefault((symbol, session), snapshot.open or snapshot.close)
        return replace(snapshot, trade_session=session, pre_close=reference,
                       open=snapshot.open or reference)

    async def stream(self):
        delay = 1.0
        while True:
            try:
                async for snapshot in self._stream_once():
                    delay = 1.0
                    yield snapshot
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self.status, self.status_reason = "retrying", f"{type(error).__name__}：长桥行情连接异常，正在重试"
                self.context = None
                print(f"{self.market} {self.status_reason}")
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)

    async def _stream_once(self):
        from longbridge.openapi import SubType
        await self.initialize()
        context = self.context
        queue = asyncio.Queue(maxsize=4096)
        loop = asyncio.get_running_loop()
        def enqueue(snapshot):
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(snapshot)
        def on_quote(symbol, event):
            snapshot = self.push_snapshot(symbol, event)
            if snapshot is not None and not loop.is_closed():
                loop.call_soon_threadsafe(enqueue, snapshot)
        context.set_on_quote(on_quote)
        subscribed = False
        try:
            try:
                await asyncio.wait_for(context.subscribe(self.api_symbols, [SubType.Quote]), 10)
                subscribed = True
                self.status_reason = "长桥推送及轮询已启用"
            except Exception:
                self.status_reason = "推送不可用，使用长桥行情轮询"
            # Poll on a fixed cadence even when another symbol keeps pushing;
            # this also detects session changes and missing extended-hours pushes.
            next_poll = loop.time()
            while True:
                if loop.time() >= next_poll:
                    for quote in await asyncio.wait_for(context.quote(self.api_symbols), 10):
                        for snapshot in self.snapshots(quote):
                            yield snapshot
                    next_poll = loop.time() + self.poll_seconds
                try:
                    yield await asyncio.wait_for(queue.get(), max(.001, next_poll - loop.time()))
                except asyncio.TimeoutError:
                    pass
        finally:
            if subscribed:
                try:
                    await asyncio.wait_for(context.unsubscribe(self.api_symbols, [SubType.Quote]), 2)
                except Exception:
                    pass


class LongbridgeHKMarketSource(LongbridgeMarketSource):
    pass


class LongbridgeUSMarketSource(LongbridgeMarketSource):
    def __init__(self, symbols, poll_seconds=2.0):
        super().__init__(symbols, poll_seconds, market="US")

    @staticmethod
    def _normalize_symbol(symbol):
        return symbol.strip().upper().removesuffix(".US")
