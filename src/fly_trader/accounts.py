"""Local, long-only simulation ledger. No broker account or trading clients."""
from __future__ import annotations

import math
from datetime import datetime, timezone

from .calendar import TradingCalendar

SIMULATION_VERSION = "track-pools-v2"
INITIAL_CASH = 1_000_000.0


class LocalAccount:
    def __init__(self, market: str, fee_bps: float = 3, slippage_bps: float = 5):
        self.market = market
        self.currency = "HKD" if market == "HK" else "USD"
        self.cash = INITIAL_CASH
        self.friction = (fee_bps + slippage_bps) / 10_000
        self.positions: dict[str, dict] = {}
        self.prices: dict[str, dict] = {}
        self.trades: list[dict] = []
        self.executed: set[str] = set()
        self.statistics: dict[str, dict] = {}
        self.realized = 0.0
        self.peak = INITIAL_CASH
        self.max_drawdown = 0.0
        self.day: str | None = None
        self.day_start_equity = INITIAL_CASH
        self.calendar = TradingCalendar()

    @property
    def equity(self) -> float:
        return self.cash + sum(p["quantity"] * self.prices[s]["price"]
                               for s, p in self.positions.items())

    def mark(self, quote) -> None:
        if not math.isfinite(quote.close) or quote.close <= 0:
            return
        day = self.calendar.local_time(quote.symbol, quote.market_time_ms).date().isoformat()
        if self.day is None or day > self.day:
            self.day_start_equity = self.equity
            self.day = day
        self.prices[quote.symbol] = {"price": quote.close,
                                     "market_time_ms": quote.market_time_ms}
        self._measure()

    def _measure(self) -> None:
        value = self.equity
        self.peak = max(self.peak, value)
        self.max_drawdown = min(self.max_drawdown, value / self.peak - 1)

    def execute(self, proposal: dict, decision_id: str, timestamp: str) -> dict | None:
        if proposal.get("status") != "ready" or decision_id in self.executed:
            return None
        symbol, side = proposal["symbol"], proposal["side"]
        quantity, price = float(proposal["quantity"]), float(proposal["reference_price"])
        if (symbol not in self.prices or side not in {"BUY", "SELL"}
                or not all(math.isfinite(x) and x > 0 for x in (quantity, price))):
            return None
        position = self.positions.get(symbol, {"quantity": 0.0, "cost": 0.0})
        gross = quantity * price
        cost = gross * self.friction
        if side == "BUY":
            if gross + cost > self.cash:
                return None
            self.cash -= gross + cost
            position = {"quantity": position["quantity"] + quantity,
                        "cost": position["cost"] + gross + cost}
            self.positions[symbol] = position
            realized = 0.0
        else:
            if quantity > position["quantity"]:
                return None
            basis = position["cost"] * quantity / position["quantity"]
            realized = gross - cost - basis
            self.realized += realized
            self.cash += gross - cost
            position = {"quantity": position["quantity"] - quantity,
                        "cost": position["cost"] - basis}
            if position["quantity"] > 0:
                self.positions[symbol] = position
            else:
                self.positions.pop(symbol, None)
        self.executed.add(decision_id)
        stats = self.statistics.setdefault(symbol, {
            "bought_quantity": 0.0, "sold_quantity": 0.0,
            "buy_amount": 0.0, "sell_amount": 0.0, "costs": 0.0,
            "realized_pl": 0.0, "cash_flow": 0.0, "trade_count": 0,
        })
        stats["bought_quantity" if side == "BUY" else "sold_quantity"] += quantity
        stats["buy_amount" if side == "BUY" else "sell_amount"] += gross
        stats["costs"] += cost
        stats["realized_pl"] += realized
        stats["cash_flow"] += -(gross + cost) if side == "BUY" else gross - cost
        stats["trade_count"] += 1
        trade = {"decision_id": decision_id, "symbol": symbol, "side": side,
                 "quantity": quantity, "reference_price": price, "costs": cost,
                 "realized_pl": realized, "currency": self.currency, "timestamp": timestamp}
        self.trades.append(trade)
        self._measure()
        return trade

    def snapshot(self, now_ms: int | None = None) -> dict:
        now_ms = now_ms if now_ms is not None else int(datetime.now(timezone.utc).timestamp() * 1000)
        equity = self.equity
        positions = []
        for symbol, p in self.positions.items():
            quote = self.prices[symbol]
            value = p["quantity"] * quote["price"]
            positions.append({
                "symbol": symbol, "name": symbol, "market": self.market,
                "currency": self.currency, "quantity": p["quantity"],
                "cost_price": p["cost"] / p["quantity"], "current_price": quote["price"],
                "market_value": value, "position_weight": value / equity if equity else 0,
                "unrealized_pl": value - p["cost"],
                "unrealized_pl_ratio": value / p["cost"] - 1 if p["cost"] else None,
                "valuation_time_ms": quote["market_time_ms"],
                "stale": now_ms - quote["market_time_ms"] > 5000,
            })
        by_symbol = {p["symbol"]: p for p in positions}
        records = []
        for symbol in dict.fromkeys([*self.prices, *self.statistics]):
            stats = self.statistics.get(symbol, {})
            position = by_symbol.get(symbol, {})
            realized = stats.get("realized_pl", 0)
            unrealized = position.get("unrealized_pl", 0)
            records.append({
                "symbol": symbol, **stats,
                "bought_quantity": stats.get("bought_quantity", 0),
                "sold_quantity": stats.get("sold_quantity", 0),
                "quantity": position.get("quantity", 0),
                "cost_price": position.get("cost_price"),
                "cost_basis": self.positions.get(symbol, {}).get("cost", 0),
                "average_buy_price": (stats["buy_amount"] / stats["bought_quantity"]
                                      if stats.get("bought_quantity") else None),
                "average_sell_price": (stats["sell_amount"] / stats["sold_quantity"]
                                       if stats.get("sold_quantity") else None),
                "current_price": self.prices.get(symbol, {}).get("price"),
                "market_value": position.get("market_value", 0),
                "realized_pl": realized, "unrealized_pl": unrealized,
                "total_pl": realized + unrealized,
            })
        return {
            "provider": "本地模拟资金池", "environment": "local_simulation", "status": "ok",
            "market": self.market, "currency": self.currency, "initial_cash": INITIAL_CASH,
            "cash": self.cash, "buying_power": self.cash, "equity": equity,
            "market_value": equity - self.cash, "day_start_equity": self.day_start_equity,
            "fx_rates": {self.currency: 1.0}, "positions": positions,
            "realized_pl": self.realized,
            "unrealized_pl": sum(p["unrealized_pl"] for p in positions),
            "return": equity / INITIAL_CASH - 1, "max_drawdown": self.max_drawdown,
            "trade_count": len(self.trades), "valuation_status": "ok",
            "bought_quantity": sum(s["bought_quantity"] for s in self.statistics.values()),
            "sold_quantity": sum(s["sold_quantity"] for s in self.statistics.values()),
            "costs": sum(s["costs"] for s in self.statistics.values()),
            "total_pl": equity - INITIAL_CASH, "symbol_records": records,
        }
