from __future__ import annotations

from datetime import datetime, time as clock_time, timezone, timedelta
from importlib.resources import files
import json
from zoneinfo import ZoneInfo


class TradingCalendar:
    """Exchange session calendar with bundled holidays and explicit fail-closed coverage."""

    def __init__(self) -> None:
        path = files("fly_trader").joinpath("calendars/trading_holidays.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        self.holidays = {
            market: frozenset(values) for market, values in data["holidays"].items()
        }
        self.coverage = data["coverage"]
        self.early_closes = data.get("early_closes", {})

    @staticmethod
    def market(symbol: str) -> str:
        return "HK" if symbol.endswith(".HK") else "US"

    def local_time(self, symbol: str, market_time_ms: int) -> datetime:
        market = self.market(symbol)
        zone = ZoneInfo("Asia/Hong_Kong" if market == "HK" else "America/New_York")
        return datetime.fromtimestamp(market_time_ms / 1000, timezone.utc).astimezone(zone)

    def session_reason(self, symbol: str, market_time_ms: int) -> str | None:
        market = self.market(symbol)
        local = self.local_time(symbol, market_time_ms)
        if not self.coverage[market]["start"] <= local.date().isoformat() <= self.coverage[market]["end"]:
            return "交易日历超出已验证范围，禁止生成订单提案"
        if local.weekday() >= 5 or local.date().isoformat() in self.holidays[market]:
            return "交易所休市日，禁止生成订单提案"
        current = local.timetz().replace(tzinfo=None)
        early = self.early_closes.get(market, {}).get(local.date().isoformat())
        if early:
            hour, minute = (int(value) for value in early.split(":"))
            if current >= clock_time(hour, minute):
                return "交易所提前收市，禁止生成订单提案"
        if market == "US":
            return None if clock_time(9, 30) <= current < clock_time(16) else "不在美股常规交易时段"
        return (None if (clock_time(9, 30) <= current < clock_time(12) or
                         clock_time(13) <= current < clock_time(16))
                else "不在港股连续交易时段")

    def us_session(self, market_time_ms: int) -> str:
        local = self.local_time("US", market_time_ms)
        minute = local.hour * 60 + local.minute
        trade_date = local.date() + timedelta(days=1 if minute >= 1200 else 0)
        date = trade_date.isoformat()
        if (not self.coverage["US"]["start"] <= date <= self.coverage["US"]["end"]
                or trade_date.weekday() >= 5 or date in self.holidays["US"]):
            return "closed"
        if minute < 240 or minute >= 1200:
            return "overnight"
        if minute < 570:
            return "pre"
        early = self.early_closes.get("US", {}).get(date)
        close = sum(int(v) * multiplier for v, multiplier in zip(early.split(":"), (60, 1))) if early else 960
        if minute < close:
            return "regular"
        if minute < (1020 if early else 1200):
            return "post"
        return "closed"

    def simulation_session_reason(self, symbol: str, market_time_ms: int) -> str | None:
        if self.market(symbol) == "HK":
            return self.session_reason(symbol, market_time_ms)
        return "当前不在美股开放的交易时段或交易日历范围外" if self.us_session(market_time_ms) == "closed" else None

    def market_data_hint(self, symbol: str, market_time_ms: int) -> str:
        """Explain the configured feeds' coverage; this does not authorize trades."""
        local = self.local_time(symbol, market_time_ms)
        market = self.market(symbol)
        if market == "US":
            session = self.us_session(market_time_ms)
            label = {"pre": "盘前", "regular": "常规", "post": "盘后", "overnight": "夜盘", "closed": "休市"}[session]
            return f"当前美东 {local:%H:%M} · {label}；等待长桥该时段有效行情，请检查连接与行情权限"
        elif self.session_reason(symbol, market_time_ms):
            return "当前不在港股连续交易时段，可能没有新行情；请等待有效报价"
        return "尚未收到有效行情，请检查行情连接、股票代码及订阅权限"
