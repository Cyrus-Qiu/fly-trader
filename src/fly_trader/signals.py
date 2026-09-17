from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import math

from .neural import NeuralReadout


@dataclass(frozen=True)
class Signal:
    action: str
    score: float
    reason: str


class SignalDecoder:
    def __init__(self, threshold: float = 8.0):
        self.threshold = threshold

    def decode(self, readout: NeuralReadout, halted: bool) -> Signal:
        if halted:
            return Signal("HOLD", readout.score, "halted quote")
        if readout.score >= self.threshold:
            return Signal("BUY", readout.score, "right-turn DN rate exceeds left-turn DN rate")
        if readout.score <= -self.threshold:
            return Signal("SELL", readout.score, "left-turn DN rate exceeds right-turn DN rate")
        return Signal("HOLD", readout.score, "directional neural score inside dead zone")


class NeuralSignalFilter:
    """EMA/hysteresis decoder with optional per-symbol percentile calibration."""

    def __init__(self, hz: int = 50, window_s: float = 1.0,
                 ema_half_life_s: float = 0.35, enter_threshold: float = 20.0,
                 exit_threshold: float = 8.0, cooldown_s: float = 1.0,
                 adaptive: bool = False, calibration_s: float = 300.0,
                 baseline_s: float = 1800.0, baseline_min_samples: int = 100,
                 enter_percentile: float = 0.95,
                 min_enter_threshold: float = 2.0,
                 max_enter_threshold: float = 20.0):
        self.hz = hz
        self.window_s = window_s
        self.ema_half_life_s = ema_half_life_s
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.cooldown_s = cooldown_s
        self.adaptive = adaptive
        self.calibration_s = calibration_s
        self.baseline_s = baseline_s
        self.baseline_min_samples = baseline_min_samples
        self.enter_percentile = enter_percentile
        self.min_enter_threshold = min_enter_threshold
        self.max_enter_threshold = max_enter_threshold
        self.scores: deque[float] = deque(maxlen=max(1, round(hz * window_s)))
        self.baseline: deque[tuple[float, float]] = deque()
        self.ema_score = 0.0
        self.window_mean = 0.0
        self.confirmed = "HOLD"
        self.last_change = -float("inf")

    def update(self, raw: Signal, now: float, stale: bool = False) -> Signal:
        self.scores.append(raw.score)
        self.window_mean = sum(self.scores) / len(self.scores)
        alpha = 1.0 - math.exp(-math.log(2.0) / (self.hz * self.ema_half_life_s))
        self.ema_score += alpha * (self.window_mean - self.ema_score)

        if not stale and math.isfinite(self.ema_score):
            self.baseline.append((now, abs(self.ema_score)))
            cutoff = now - self.baseline_s
            while self.baseline and self.baseline[0][0] < cutoff:
                self.baseline.popleft()
        calibrated = (not self.adaptive or
                      (now >= self.calibration_s and
                       len(self.baseline) >= self.baseline_min_samples))
        if self.adaptive and self.baseline:
            ordered = sorted(value for _, value in self.baseline)
            index = min(len(ordered) - 1,
                        max(0, math.ceil(self.enter_percentile * len(ordered)) - 1))
            self.enter_threshold = min(self.max_enter_threshold,
                                       max(self.min_enter_threshold, ordered[index]))
            self.exit_threshold = self.enter_threshold * 0.4

        if stale:
            if self.confirmed != "HOLD":
                self.last_change = now
            self.confirmed = "HOLD"
            return Signal("HOLD", self.ema_score, "market data older than 5 seconds")

        if not calibrated:
            self.confirmed = "HOLD"
            return Signal("HOLD", self.ema_score, "adaptive threshold calibration")

        target = self.confirmed
        if self.confirmed == "HOLD":
            if self.ema_score >= self.enter_threshold:
                target = "BUY"
            elif self.ema_score <= -self.enter_threshold:
                target = "SELL"
        elif self.confirmed == "BUY" and self.ema_score < self.exit_threshold:
            target = "HOLD"
        elif self.confirmed == "SELL" and self.ema_score > -self.exit_threshold:
            target = "HOLD"

        if target != self.confirmed and now - self.last_change >= self.cooldown_s:
            self.confirmed = target
            self.last_change = now
            reason = "hysteresis threshold crossed"
        elif target != self.confirmed:
            reason = "direction blocked by cooldown"
        else:
            reason = "hysteresis state retained"
        return Signal(self.confirmed, self.ema_score, reason)

    def diagnostics(self) -> dict[str, float | int | bool | str]:
        calibrated = (not self.adaptive or
                      (self.baseline and self.baseline[-1][0] >= self.calibration_s and
                       len(self.baseline) >= self.baseline_min_samples))
        return {
            "window_s": self.window_s,
            "samples": len(self.scores),
            "window_mean_score": self.window_mean,
            "ema_half_life_s": self.ema_half_life_s,
            "ema_score": self.ema_score,
            "enter_threshold": self.enter_threshold,
            "exit_threshold": self.exit_threshold,
            "cooldown_s": self.cooldown_s,
            "adaptive": self.adaptive,
            "calibration_status": "ready" if calibrated else "warming_up",
            "calibration_s": self.calibration_s,
            "warmup_remaining_s": (0.0 if not self.baseline else
                                     max(0.0, self.calibration_s - self.baseline[-1][0])),
            "baseline_samples": len(self.baseline),
            "enter_percentile": self.enter_percentile,
            "distance_to_trigger": max(0.0, self.enter_threshold - abs(self.ema_score)),
        }

    def snapshot(self) -> dict:
        return {
            "scores": list(self.scores),
            "baseline": list(self.baseline),
            "ema_score": self.ema_score,
            "window_mean": self.window_mean,
            "confirmed": self.confirmed,
            "last_change": self.last_change,
            "enter_threshold": self.enter_threshold,
            "exit_threshold": self.exit_threshold,
        }

    def restore(self, value: dict) -> None:
        try:
            self.scores.clear()
            self.scores.extend(float(item) for item in value.get("scores", []))
            self.baseline.clear()
            self.baseline.extend((float(a), float(b)) for a, b in value.get("baseline", []))
            self.ema_score = float(value["ema_score"])
            self.window_mean = float(value["window_mean"])
            confirmed = str(value["confirmed"])
            self.confirmed = confirmed if confirmed in {"BUY", "HOLD", "SELL"} else "HOLD"
            self.last_change = float(value["last_change"])
            self.enter_threshold = float(value["enter_threshold"])
            self.exit_threshold = float(value["exit_threshold"])
        except (KeyError, TypeError, ValueError):
            self.scores.clear()
            self.baseline.clear()
