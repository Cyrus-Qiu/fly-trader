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

    def mark(self, symbol: str, price: float) -> dict | None:
        """Revalue existing comparisons without introducing another signal decision."""
        portfolios = self._portfolios.get(symbol)
        if portfolios is None:
            return None
        return {
            "symbol": symbol, "points": self._points[symbol],
            "fee_bps": self.fee_bps, "slippage_bps": self.slippage_bps,
            "scope": "current_process_session",
            "tracks": {name: {"label": self.labels[name],
                               **portfolio.update(price, "HOLD", self.friction)}
                       for name, portfolio in portfolios.items()},
        }


class TrackPerformance:
    """Independent million-dollar accounts; signal tracks use equal symbol budgets."""
    labels = LivePerformanceTracker.labels

    def __init__(self, market, symbols):
        from .accounts import LocalAccount, INITIAL_CASH
        self.symbols = list(symbols)
        self.market = market
        self.accounts = {track: LocalAccount(market) for track in self.labels}
        self.symbol_budget = INITIAL_CASH / len(symbols) if symbols else 0
        self.execution_notes = {track: {} for track in self.labels}

    def mark(self, quote):
        for account in self.accounts.values():
            account.mark(quote)

    def wait_for_market(self, symbol, reason):
        for track in self.labels:
            if track != "disconnected_connectome":
                self.execution_notes[track][symbol] = reason

    def update(self, symbol, price, result, decision_id, timestamp, lots):
        import math
        actions = {"fly_raw": result.raw_signal.action,
                   "fly_filtered": result.filtered_signal.action,
                   "buy_and_hold": "BUY", "disconnected_connectome": "HOLD"}
        fills = []
        for track, account in self.accounts.items():
            if track == "risk_executable":
                proposal = result.proposal
                if not proposal or proposal.get("status") != "ready":
                    self.execution_notes[track][symbol] = proposal.get("reason", "等待行情") if proposal else "等待行情"
                    continue
            else:
                action = actions[track]
                held = account.positions.get(symbol, {}).get("quantity", 0)
                if track == "disconnected_connectome":
                    continue
                if action == "HOLD":
                    if track == "fly_filtered":
                        d = result.filter_diagnostics
                        if d.get("market_stale"):
                            note = "行情已过期，人工过滤强制观望"
                        elif d.get("calibration_status") != "ready":
                            note = (f"等待校准：剩余 {d.get('warmup_remaining_s', 0):.0f} 秒，"
                                    f"有效样本 {d.get('baseline_samples', 0)}/{d.get('baseline_min_samples', 100)}")
                        else:
                            note = "人工过滤信号为观望，尚未达到交易触发条件"
                    else:
                        note = "原始信号为观望"
                    self.execution_notes[track][symbol] = note
                    continue
                if action == "BUY" and held:
                    self.execution_notes[track][symbol] = "已有持仓，重复买入信号不加仓"
                    continue
                if action == "SELL" and not held:
                    self.execution_notes[track][symbol] = "当前空仓，无可卖股数"
                    continue
                if track == "buy_and_hold" and account.statistics.get(symbol, {}).get("bought_quantity", 0):
                    continue
                lot = lots.get(symbol) if self.market == "HK" else 1
                if not isinstance(lot, int) or lot <= 0:
                    self.execution_notes[track][symbol] = "缺少每手股数，未模拟成交"
                    continue
                budget = self.symbol_budget + account.statistics.get(symbol, {}).get("cash_flow", 0)
                quantity = (math.floor(min(account.cash, max(0, budget)) / (price * (1 + account.friction) * lot)) * lot
                            if action == "BUY" else held)
                if quantity <= 0:
                    self.execution_notes[track][symbol] = "分配资金不足一个交易单位"
                    continue
                proposal = {"status": "ready", "symbol": symbol, "side": action,
                            "quantity": quantity, "reference_price": price}
            fill = account.execute(proposal, f"{decision_id}:{track}", timestamp)
            if fill:
                fill["track"] = track
                fills.append(fill)
                self.execution_notes[track][symbol] = "已模拟成交"
            elif track == "risk_executable":
                proposal.update(status="blocked", reason="本地资金或持仓校验未通过")
        return fills

    def snapshot(self):
        result = {}
        for track, account in self.accounts.items():
            snapshot = account.snapshot()
            records = {r["symbol"]: r for r in snapshot["symbol_records"]}
            for symbol in self.symbols:
                records.setdefault(symbol, {"symbol": symbol, "bought_quantity": 0,
                    "sold_quantity": 0, "quantity": 0, "cost_price": None,
                    "average_buy_price": None, "average_sell_price": None,
                    "cost_basis": 0, "market_value": 0, "realized_pl": 0,
                    "unrealized_pl": 0, "total_pl": 0, "current_price": None})
                records[symbol]["execution_note"] = self.execution_notes[track].get(symbol, "等待信号" if track != "disconnected_connectome" else "现金对照，不交易")
            result[track] = {**snapshot, "label": self.labels[track], "track": track,
                             "symbol_records": list(records.values()),
                             "allocation": "risk_proposal" if track == "risk_executable" else "equal_symbol_budget",
                             "symbol_budget": None if track == "risk_executable" else self.symbol_budget}
        return result
