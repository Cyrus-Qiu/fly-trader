from __future__ import annotations

from datetime import datetime, time as clock_time, timezone
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
