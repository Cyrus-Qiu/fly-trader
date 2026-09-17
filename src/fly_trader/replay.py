from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from statistics import median

from .experiment import EXPERIMENT_ID, EXPERIMENT_FINGERPRINT


TRACKS = ("fly_raw", "fly_filtered", "risk_executable")


@dataclass(frozen=True)
class ReplayPoint:
    symbol: str
    market_time_ms: int
    price: float
    actions: dict[str, str]


def _legacy_tracks(event: dict) -> dict[str, dict]:
    raw = event.get("raw_signal") or event.get("signal")
    filtered = event.get("filtered_signal") or raw
    proposal = event.get("order_proposal")
    executable = {
        "action": (proposal.get("side") if proposal and proposal.get("status") == "ready"
                   else "HOLD")
    }
    return {"fly_raw": raw, "fly_filtered": filtered,
            "risk_executable": executable}


def load_events(paths: list[Path], include_legacy: bool = False
                ) -> tuple[list[ReplayPoint], dict]:
    points: list[ReplayPoint] = []
    diagnostics = {
        "files": 0, "lines": 0, "accepted_events": 0,
        "legacy_events_skipped": 0, "wrong_experiment_skipped": 0,
        "invalid_lines": 0, "unsupported_events": 0,
    }
    for path in paths:
        diagnostics["files"] += 1
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                diagnostics["lines"] += 1
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    diagnostics["invalid_lines"] += 1
                    continue
                schema = int(event.get("schema", 0) or 0)
                if schema < 8 and not include_legacy:
                    diagnostics["legacy_events_skipped"] += 1
                    continue
                if schema >= 8:
                    experiment = event.get("experiment", {})
                    if (experiment.get("experiment_id") != EXPERIMENT_ID or
                            experiment.get("fingerprint") != EXPERIMENT_FINGERPRINT):
                        diagnostics["wrong_experiment_skipped"] += 1
                        continue
                try:
                    quote = event["quote"]
                    tracks = event.get("tracks") or _legacy_tracks(event)
                    symbol = str(event.get("symbol") or quote["symbol"]).upper()
                    price = float(quote["close"])
                    timestamp = int(quote["market_time_ms"])
                    actions = {name: str(tracks[name]["action"]) for name in TRACKS}
                    if (price <= 0 or
                            any(action not in {"BUY", "HOLD", "SELL"}
                                for action in actions.values())):
                        raise ValueError("invalid point")
                    points.append(ReplayPoint(symbol, timestamp, price, actions))
                    diagnostics["accepted_events"] += 1
                except (KeyError, TypeError, ValueError):
                    diagnostics["unsupported_events"] += 1
    points.sort(key=lambda item: (item.symbol, item.market_time_ms))
    return points, diagnostics


def _max_drawdown(curve: list[float]) -> float:
    peak, worst = curve[0], 0.0
    for value in curve:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, value / peak - 1.0)
    return worst


def _simulate(prices: list[float], actions: list[str],
              fee_bps: float, slippage_bps: float) -> dict:
    initial = prices[0]
    cash, quantity, entry_value = initial, 0.0, None
    curve, trade_returns = [], []
    turnover, trades, holding_points = 0.0, 0, 0
    friction = (fee_bps + slippage_bps) / 10_000.0
    for price, action in zip(prices, actions):
        if action == "BUY" and quantity == 0:
            execution = price * (1.0 + friction)
            quantity = cash / execution
            turnover += cash / initial
            entry_value = cash
            cash = 0.0
            trades += 1
        elif action == "SELL" and quantity > 0:
            cash = quantity * price * (1.0 - friction)
            turnover += cash / initial
            if entry_value:
                trade_returns.append(cash / entry_value - 1.0)
            quantity = 0.0
            entry_value = None
            trades += 1
        if quantity > 0:
            holding_points += 1
        curve.append(cash + quantity * price)
    final = curve[-1]
    return {
        "total_return": final / initial - 1.0,
        "max_drawdown": _max_drawdown(curve),
        "trades": trades,
        "round_trips": len(trade_returns),
        "win_rate": (sum(value > 0 for value in trade_returns) / len(trade_returns)
                     if trade_returns else None),
        "mean_round_trip_return": (
            sum(trade_returns) / len(trade_returns) if trade_returns else None),
        "turnover": turnover,
        "exposure_fraction": holding_points / len(prices),
        "final_value": final,
    }


def _random_actions(actions: list[str], seed: int) -> list[str]:
    transitions = []
    holding = False
    for action in actions:
        if action == "BUY" and not holding:
            transitions.append(action)
            holding = True
        elif action == "SELL" and holding:
            transitions.append(action)
            holding = False
    result = ["HOLD"] * len(actions)
    if not transitions:
        return result
    rng = random.Random(seed)
    indexes = sorted(rng.sample(range(len(actions)), min(len(transitions), len(actions))))
    for index, action in zip(indexes, transitions):
        result[index] = action
    return result


def evaluate(paths: list[Path], fee_bps: float = 3.0,
             slippage_bps: float = 5.0, seed: int = 64,
             include_legacy: bool = False) -> dict:
    points, diagnostics = load_events(paths, include_legacy)
    grouped: dict[str, list[ReplayPoint]] = {}
    for point in points:
        grouped.setdefault(point.symbol, []).append(point)
    symbols = {}
    for symbol, values in grouped.items():
        prices = [item.price for item in values]
        actions = {
            track: [item.actions[track] for item in values] for track in TRACKS
        }
        buy_hold = ["BUY"] + ["HOLD"] * (len(values) - 1)
        disconnected = ["HOLD"] * len(values)
        gaps = [b.market_time_ms - a.market_time_ms for a, b in zip(values, values[1:])
                if b.market_time_ms > a.market_time_ms]
        typical_gap = median(gaps) if gaps else 0
        symbols[symbol] = {
            "events": len(values),
            "start_time_ms": values[0].market_time_ms,
            "end_time_ms": values[-1].market_time_ms,
            "data_gaps": sum(gap > max(60_000, typical_gap * 10) for gap in gaps),
            **{track: _simulate(prices, actions[track], fee_bps, slippage_bps)
               for track in TRACKS},
            "buy_and_hold": _simulate(prices, buy_hold, fee_bps, slippage_bps),
            "random_same_frequency": _simulate(
                prices, _random_actions(actions["fly_raw"], seed),
                fee_bps, slippage_bps),
            "disconnected_connectome": _simulate(
                prices, disconnected, fee_bps, slippage_bps),
        }
    return {
        "schema": 2,
        "mode": "frozen-dual-track-offline-evaluation",
        "experiment_id": EXPERIMENT_ID,
        "experiment_fingerprint": EXPERIMENT_FINGERPRINT,
        "primary_result": "fly_raw",
        "assumptions": {
            "initial_capital": "one initial share-price unit per symbol",
            "positioning": "long-only, all-in on BUY and flat on SELL",
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "random_seed": seed,
            "disconnected_baseline": (
                "expected-value zero directional exposure; not a stochastic rerun"),
            "legacy_events_included": include_legacy,
        },
        "diagnostics": diagnostics,
        "symbols": symbols,
    }


def write_report(report: dict, path: Path | None = None) -> str:
    body = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body + "\n", encoding="utf-8")
    return body
