from __future__ import annotations

from dataclasses import dataclass
import math
import time

from .calendar import TradingCalendar
from .market import MarketSnapshot
from .signals import Signal


@dataclass
class _SignalState:
    action: str = "HOLD"
    since: float = 0.0
    last_ready_at: float | None = None


class OrderProposalEngine:
    """Pure read-only proposal/risk preview. It has no broker or order methods."""

    allowed_symbols = frozenset({"NVDA", "700.HK", "2513.HK"})
    lot_sizes = {"NVDA": 1, "700.HK": 100, "2513.HK": 100}

    def __init__(self, confirmation_s: float = 3.0,
                 max_symbol_weight: float = 0.10,
                 max_total_weight: float = 0.70,
                 proposal_cooldown_s: float = 300.0,
                 max_daily_loss: float = 0.02) -> None:
        self.confirmation_s = confirmation_s
        self.max_symbol_weight = max_symbol_weight
        self.max_total_weight = max_total_weight
        self.proposal_cooldown_s = proposal_cooldown_s
        self.max_daily_loss = max_daily_loss
        self._states: dict[str, _SignalState] = {}
        self._day_start_equity: dict[tuple[str, str], float] = {}
        self.calendar = TradingCalendar()

    def snapshot(self) -> dict:
        return {
            "states": {
                symbol: {
                    "action": state.action,
                    "since": state.since,
                    "last_ready_at": state.last_ready_at,
                } for symbol, state in self._states.items()
            },
            "day_start_equity": [
                {"account": account, "session": session, "equity": equity}
                for (account, session), equity in self._day_start_equity.items()
            ],
        }

    def restore(self, value: dict) -> None:
        try:
            self._states = {
                symbol: _SignalState(
                    action=str(state.get("action", "HOLD")),
                    since=float(state.get("since", 0)),
                    last_ready_at=(None if state.get("last_ready_at") is None
                                   else float(state["last_ready_at"])),
                )
                for symbol, state in value.get("states", {}).items()
            }
            self._day_start_equity = {
                (str(item["account"]), str(item["session"])): float(item["equity"])
                for item in value.get("day_start_equity", [])
            }
        except (KeyError, TypeError, ValueError):
            self._states = {}
            self._day_start_equity = {}

    @staticmethod
    def _checks() -> dict[str, dict]:
        labels = (
            ("allowlist", "交易白名单"), ("signal", "稳定信号"),
            ("confirmation", "信号确认"), ("quote", "行情质量"),
            ("session", "交易时段"), ("account", "模拟账户"),
            ("daily_loss", "日损失熔断"), ("long_only", "只做多"),
            ("cash", "现金限制"), ("symbol_weight", "单标的仓位"),
            ("total_weight", "总仓位"), ("cooldown", "提案冷却"),
        )
        return {name: {"name": name, "label": label, "status": "pending", "reason": "待检查"}
                for name, label in labels}

    @staticmethod
    def _set(checks: dict[str, dict], name: str, status: str, reason: str) -> None:
        checks[name].update(status=status, reason=reason)

    @staticmethod
    def _skip_remaining(checks: dict[str, dict]) -> None:
        for check in checks.values():
            if check["status"] == "pending":
                check.update(status="skipped", reason="前置条件未通过")

    def evaluate(self, symbol: str, signal: Signal, quote: MarketSnapshot,
                 account: dict | None, quote_age_s: float = 0.0,
                 now: float | None = None) -> dict:
        now = time.monotonic() if now is None else now
        state = self._states.setdefault(symbol, _SignalState(since=now))
        if signal.action != state.action:
            state.action, state.since = signal.action, now
        held_s = max(0.0, now - state.since)
        checks = self._checks()
        base = {
            "symbol": symbol, "action": signal.action, "score": signal.score,
            "status": "observing", "reason": "稳定信号为观望",
            "quantity": 0, "reference_price": quote.close,
            "signal_held_s": held_s, "confirmation_s": self.confirmation_s,
            "proposal_cooldown_s": self.proposal_cooldown_s,
            "order_submission_enabled": False,
        }

        def result(status: str, reason: str, **values) -> dict:
            if status != "ready":
                self._skip_remaining(checks)
            return {**base, "status": status, "reason": reason,
                    "risk_checks": list(checks.values()), **values}

        if symbol not in self.allowed_symbols:
            self._set(checks, "allowlist", "blocked", "标的不在交易白名单")
            return result("blocked", "标的不在交易白名单")
        self._set(checks, "allowlist", "passed", "标的在硬编码白名单内")
        if signal.action == "HOLD":
            self._set(checks, "signal", "observing", "稳定信号为观望")
            return result("observing", "稳定信号为观望")
        self._set(checks, "signal", "passed", f"稳定信号为 {signal.action}")
        if held_s < self.confirmation_s:
            self._set(checks, "confirmation", "confirming",
                      f"已持续 {held_s:.1f}/{self.confirmation_s:g} 秒")
            return result("confirming", f"等待信号持续 {self.confirmation_s:g} 秒")
        self._set(checks, "confirmation", "passed", "信号持续时间达标")
        if quote_age_s > 5.0 or "bmp" in quote.feed:
            self._set(checks, "quote", "blocked", "报价过期或为延迟行情")
            return result("blocked", "报价过期或为延迟行情")
        if quote.feed == "overnight":
            self._set(checks, "quote", "blocked", "隔夜指示性报价不可用于订单提案")
            return result("blocked", "隔夜指示性报价不可用于订单提案")
        self._set(checks, "quote", "passed", "行情新鲜且可用于提案")
        session_reason = self.calendar.session_reason(symbol, quote.market_time_ms)
        if session_reason:
            self._set(checks, "session", "blocked", session_reason)
            return result("blocked", session_reason)
        self._set(checks, "session", "passed", "处于常规连续交易时段")
        if not account or account.get("status") != "ok":
            self._set(checks, "account", "blocked", "模拟账户数据尚未就绪")
            return result("blocked", "模拟账户数据尚未就绪")
        self._set(checks, "account", "passed", "模拟账户快照可用")

        market = self.calendar.market(symbol)
        local_time = self.calendar.local_time(symbol, quote.market_time_ms)
        equity = float(account.get("equity", 0))
        account_key = str(account.get("provider", "paper"))
        baseline_key = (account_key, f"{market}:{local_time.date().isoformat()}")
        configured_baseline = float(account.get("day_start_equity", 0) or 0)
        if configured_baseline > 0:
            self._day_start_equity[baseline_key] = configured_baseline
        elif equity > 0:
            self._day_start_equity.setdefault(baseline_key, equity)
        baseline = self._day_start_equity.get(baseline_key, 0.0)
        daily_return = equity / baseline - 1.0 if baseline > 0 else math.nan
        if not math.isfinite(daily_return):
            self._set(checks, "daily_loss", "blocked", "缺少日初净值，无法计算日损失")
            return result("blocked", "缺少日初净值，无法计算日损失")
        if daily_return <= -self.max_daily_loss:
            reason = f"账户日内损失达到 {abs(daily_return):.2%}，触发熔断"
            self._set(checks, "daily_loss", "blocked", reason)
            return result("blocked", reason, daily_return=daily_return)
        self._set(checks, "daily_loss", "passed", f"账户日内变动 {daily_return:.2%}")

        positions = account.get("positions", [])
        position = next((item for item in positions if item.get("symbol") == symbol), None)
        current_quantity = float(position.get("quantity", 0)) if position else 0.0
        lot = self.lot_sizes[symbol]
        if signal.action == "SELL":
            if current_quantity <= 0:
                self._set(checks, "long_only", "blocked", "无可卖持仓；禁止裸卖空")
                return result("blocked", "无可卖持仓；禁止裸卖空")
            self._set(checks, "long_only", "passed", "卖出数量不超过现有持仓")
            for name in ("cash", "symbol_weight", "total_weight"):
                self._set(checks, name, "skipped", "卖出提案无需检查")
            quantity = min(float(lot), current_quantity)
        else:
            self._set(checks, "long_only", "passed", "买入不会形成空头仓位")
            currency = "HKD" if symbol.endswith(".HK") else "USD"
            fx = float(account.get("fx_rates", {}).get(currency, math.nan))
            cash = float(account.get("cash", 0))
            if not math.isfinite(fx) or equity <= 0:
                self._set(checks, "cash", "blocked", "缺少账户汇率或净资产")
                return result("blocked", "缺少账户汇率或净资产")
            order_value = quote.close * lot * fx
            current_weight = float(position.get("position_weight") or 0) if position else 0.0
            total_weight = sum(float(item.get("position_weight") or 0) for item in positions)
            projected_symbol = current_weight + order_value / equity
            projected_total = total_weight + order_value / equity
            if order_value > cash:
                self._set(checks, "cash", "blocked", "现金不足；禁止融资")
                return result("blocked", "现金不足；禁止融资", quantity=lot, side="BUY")
            self._set(checks, "cash", "passed", "可用现金足够且不使用融资")
            if projected_symbol > self.max_symbol_weight:
                self._set(checks, "symbol_weight", "blocked", "预计单股仓位超过 10%")
                return result("blocked", "预计单股仓位超过 10%", quantity=lot,
                              side="BUY", projected_weight=projected_symbol)
            self._set(checks, "symbol_weight", "passed", f"预计仓位 {projected_symbol:.2%}")
            if projected_total > self.max_total_weight:
                self._set(checks, "total_weight", "blocked", "预计总仓位超过 70%")
                return result("blocked", "预计总仓位超过 70%", quantity=lot,
                              side="BUY", projected_weight=projected_symbol)
            self._set(checks, "total_weight", "passed", f"预计总仓位 {projected_total:.2%}")
            quantity = float(lot)

        if state.last_ready_at is not None:
            remaining = self.proposal_cooldown_s - (now - state.last_ready_at)
            if remaining > 0:
                self._set(checks, "cooldown", "blocked", f"提案冷却剩余 {remaining:.1f} 秒")
                return result("blocked", f"提案冷却剩余 {remaining:.1f} 秒",
                              quantity=quantity, side=signal.action,
                              cooldown_remaining_s=remaining)
        self._set(checks, "cooldown", "passed", "未处于提案冷却期")
        state.last_ready_at = now
        values = {"quantity": quantity, "side": signal.action,
                  "daily_return": daily_return}
        if signal.action == "BUY":
            values.update(order_value_account_currency=order_value,
                          projected_weight=projected_symbol)
        return result("ready", "通过完整只读风控预演", **values)
