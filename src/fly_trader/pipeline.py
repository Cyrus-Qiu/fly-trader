from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .market import MarketSnapshot
from .neural import NeuralReadout, Stimulus
from .proposals import OrderProposalEngine
from .signals import NeuralSignalFilter, Signal, SignalDecoder
from .experiment import experiment_identity


@dataclass(frozen=True)
class PipelineResult:
    """One deterministic strategy decision produced from a neural readout."""

    raw_signal: Signal
    filtered_signal: Signal
    proposal: dict | None
    filter_diagnostics: dict

    @property
    def executable_signal(self) -> Signal:
        if self.proposal and self.proposal.get("status") == "ready":
            return Signal(str(self.proposal.get("side", "HOLD")),
                          self.filtered_signal.score,
                          "通过只读风控，可执行轨道接受该动作")
        reason = ("无行情，无法执行" if self.proposal is None else
                  str(self.proposal.get("reason", "未通过只读风控")))
        return Signal("HOLD", self.filtered_signal.score, reason)


class StrategyPipeline:
    """Shared signal and risk path for live feeds and future event replay."""

    def __init__(self, hz: int, *, baseline_min_samples: int = 100,
                 proposal_engine: OrderProposalEngine | None = None) -> None:
        self.decoder = SignalDecoder()
        self.filter = NeuralSignalFilter(
            hz=hz, window_s=1.0, ema_half_life_s=0.35,
            enter_threshold=20.0, exit_threshold=8.0, cooldown_s=1.0,
            adaptive=True, baseline_min_samples=baseline_min_samples,
        )
        self.proposal_engine = proposal_engine or OrderProposalEngine()
        self.elapsed_offset = 0.0
        self.last_elapsed = 0.0

    def snapshot(self) -> dict:
        return {"filter": self.filter.snapshot(), "elapsed": self.last_elapsed}

    def restore(self, value: dict) -> None:
        self.filter.restore(value.get("filter", {}))
        self.elapsed_offset = float(value.get("elapsed", 0) or 0)

    def evaluate(self, activity: NeuralReadout, quote: MarketSnapshot | None,
                 elapsed_s: float, account: dict | None,
                 quote_age_s: float = 0.0) -> PipelineResult:
        logical_elapsed = self.elapsed_offset + elapsed_s
        self.last_elapsed = logical_elapsed
        raw = self.decoder.decode(activity, bool(quote.halted) if quote else False)
        filtered = self.filter.update(
            raw, logical_elapsed, stale=quote is None or quote_age_s >= 5.0)
        proposal = None
        if quote is not None:
            proposal = self.proposal_engine.evaluate(
                quote.symbol, filtered, quote, account, quote_age_s, logical_elapsed)
        return PipelineResult(raw, filtered, proposal, self.filter.diagnostics())

    @staticmethod
    def event(*, quote: MarketSnapshot, stimulus: Stimulus,
              activity: NeuralReadout, result: PipelineResult,
              neural_hz: int, model_step: int,
              model_dt_s: float, tick_latency_ms: float = 0.0,
              quote_age_s: float = 0.0, stimulus_scale: float = 1.0,
              elapsed_s: float | None = None,
              reference_price: float | None = None,
              extra: dict[str, Any] | None = None) -> dict:
        """Build the versioned event envelope shared by every market adapter."""
        event = {
            "schema": 8,
            "mode": "order-proposal-preview-no-orders",
            "symbol": quote.symbol,
            "feed": quote.feed,
            "event_type": quote.event_type,
            "neural_hz": neural_hz,
            "model_step": model_step,
            "model_dt_s": model_dt_s,
            "tick_latency_ms": tick_latency_ms,
            "quote": quote.public_dict(),
            "stimulus": asdict(stimulus),
            "neural_activity": asdict(activity),
            "raw_signal": asdict(result.raw_signal),
            "filtered_signal": asdict(result.filtered_signal),
            "tracks": {
                "fly_raw": asdict(result.raw_signal),
                "fly_filtered": asdict(result.filtered_signal),
                "risk_executable": asdict(result.executable_signal),
            },
            "experiment": experiment_identity(),
            "order_proposal": result.proposal,
            "signal_filter": result.filter_diagnostics,
            "market_freshness": {
                "quote_age_s": quote_age_s,
                "stimulus_scale": stimulus_scale,
                "forced_hold": quote_age_s >= 5.0,
            },
        }
        if elapsed_s is not None:
            event["evaluation"] = {
                "elapsed_s": elapsed_s,
                "reference_price": reference_price,
                "return_since_start": (
                    quote.close / reference_price - 1.0 if reference_price else 0.0
                ),
            }
        if extra:
            event.update(extra)
        return event

    @staticmethod
    def transition_event(*, quote: MarketSnapshot, activity: NeuralReadout,
                         result: PipelineResult, neural_hz: int, model_step: int,
                         quote_age_s: float, stimulus_scale: float) -> dict:
        """Compact decision-transition record; preserves science without log bloat."""
        return {
            "schema": 8,
            "mode": "fly-core-neural-transition-no-orders",
            "symbol": quote.symbol,
            "feed": quote.feed,
            "event_type": "neural_transition",
            "decision_trigger": "neural_transition",
            "neural_hz": neural_hz,
            "model_step": model_step,
            "quote": {
                "symbol": quote.symbol,
                "market_time_ms": quote.market_time_ms,
                "close": quote.close,
            },
            "neural_activity": {
                "buy_rate": activity.buy_rate,
                "sell_rate": activity.sell_rate,
                "score": activity.score,
            },
            "tracks": {
                "fly_raw": asdict(result.raw_signal),
                "fly_filtered": asdict(result.filtered_signal),
                "risk_executable": asdict(result.executable_signal),
            },
            "experiment": experiment_identity(),
            "market_freshness": {
                "quote_age_s": quote_age_s,
                "stimulus_scale": stimulus_scale,
                "forced_hold": quote_age_s >= 5.0,
            },
        }
