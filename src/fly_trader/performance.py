from __future__ import annotations

from dataclasses import dataclass

from .pipeline import PipelineResult


@dataclass
class _Portfolio:
    cash: float = 1.0
    quantity: float = 0.0
    peak: float = 1.0
    max_drawdown: float = 0.0
    trades: int = 0
    round_trips: int = 0
    wins: int = 0
    entry_value: float | None = None

    def update(self, price: float, action: str, friction: float) -> dict:
        if action == "BUY" and self.quantity == 0:
            self.quantity = self.cash / (price * (1 + friction))
            self.entry_value = self.cash
            self.cash = 0.0
            self.trades += 1
        elif action == "SELL" and self.quantity > 0:
            self.cash = self.quantity * price * (1 - friction)
            if self.entry_value is not None:
                self.round_trips += 1
                self.wins += int(self.cash > self.entry_value)
            self.quantity = 0.0
            self.entry_value = None
            self.trades += 1
        value = self.cash + self.quantity * price
        self.peak = max(self.peak, value)
        if self.peak > 0:
            self.max_drawdown = min(self.max_drawdown, value / self.peak - 1)
        return {
            "return": value - 1.0,
            "max_drawdown": self.max_drawdown,
            "trades": self.trades,
            "round_trips": self.round_trips,
            "win_rate": self.wins / self.round_trips if self.round_trips else None,
            "holding": self.quantity > 0,
            "value": value,
        }


class LivePerformanceTracker:
    """Session-local hypothetical P&L using the same execution model as replay."""

    labels = {
        "fly_raw": "果蝇原始",
        "fly_filtered": "人工过滤",
        "risk_executable": "风控可执行",
        "buy_and_hold": "买入持有",
        "disconnected_connectome": "断开连接组",
    }

    def __init__(self, fee_bps: float = 3.0, slippage_bps: float = 5.0) -> None:
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.friction = (fee_bps + slippage_bps) / 10_000
        self._portfolios: dict[str, dict[str, _Portfolio]] = {}
        self._points: dict[str, int] = {}

    def update(self, symbol: str, price: float, result: PipelineResult) -> dict:
        portfolios = self._portfolios.setdefault(
            symbol, {name: _Portfolio() for name in self.labels})
        self._points[symbol] = self._points.get(symbol, 0) + 1
        actions = {
            "fly_raw": result.raw_signal.action,
            "fly_filtered": result.filtered_signal.action,
            "risk_executable": result.executable_signal.action,
            "buy_and_hold": "BUY",
            "disconnected_connectome": "HOLD",
        }
        tracks = {}
        for name, portfolio in portfolios.items():
            tracks[name] = {
                "label": self.labels[name],
                **portfolio.update(price, actions[name], self.friction),
            }
        return {
            "symbol": symbol,
            "points": self._points[symbol],
            "fee_bps": self.fee_bps,
            "slippage_bps": self.slippage_bps,
            "scope": "current_process_session",
            "tracks": tracks,
        }
