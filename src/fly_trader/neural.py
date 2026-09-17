from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import json
import os

import numpy as np
from scipy import sparse

from .market import MarketSnapshot


@dataclass(frozen=True)
class Stimulus:
    return_from_previous_close: float
    return_from_open: float
    range_position: float
    range_width: float
    log_volume_delta: float
    halted: float
    visual_drive_mean: float


@dataclass(frozen=True)
class NeuralReadout:
    buy_rate: float
    sell_rate: float
    arousal_rate: float
    score: float
    spikes: int


class MarketEncoder:
    """Map quote features to deterministic currents across MaleCNS photoreceptors."""

    def __init__(self, visual_count: int):
        self.visual_count = visual_count
        self.previous_volume: dict[str, float] = {}
        # Six contiguous receptor bands, stable across runs/cache ordering.
        self.bands = np.array_split(np.arange(visual_count), 6)

    @staticmethod
    def _bounded_return(a: float, b: float) -> float:
        if b <= 0:
            return 0.0
        return float(np.clip(np.log(max(a, 1e-12) / b) / 0.02, -1, 1))

    def encode(self, quote: MarketSnapshot) -> tuple[np.ndarray, Stimulus]:
        prev_volume = self.previous_volume.get(quote.symbol, quote.volume)
        self.previous_volume[quote.symbol] = quote.volume
        pc = self._bounded_return(quote.close, quote.pre_close)
        op = self._bounded_return(quote.close, quote.open)
        width = max(quote.high - quote.low, 0.0)
        pos = 0.0 if width == 0 else float(np.clip(2 * (quote.close - quote.low) / width - 1, -1, 1))
        range_width = 0.0 if quote.pre_close <= 0 else float(np.clip(width / quote.pre_close / 0.04, 0, 1))
        volume_delta = float(np.clip(np.log1p(quote.volume) - np.log1p(prev_volume), -1, 1))
        halted = 1.0 if quote.halted else 0.0
        features = (pc, -pc, op, -op, pos, range_width + abs(volume_delta))
        drive = np.zeros(self.visual_count, dtype=np.float32)
        for band, value in zip(self.bands, features):
            drive[band] = np.clip(0.5 + 0.5 * value, 0, 1)
        if halted:
            drive.fill(0.0)
        return drive, Stimulus(pc, op, pos, range_width, volume_delta, halted, float(drive.mean()))


class MaleCNSConnectome:
    """Immutable graph data shared by every per-symbol neural state."""

    def __init__(self, cache: Path):
        if not (cache / "manifest.json").exists():
            raise FileNotFoundError(f"MaleCNS cache missing: {cache}")
        meta = np.load(cache / "model.npz", allow_pickle=False)
        self.n = int(meta["n"])
        self.w = sparse.load_npz(cache / "weights.npz").astype(np.float32).tocsc()
        self.visual = meta["visual"]
        self.forward = meta["forward"]
        self.turn_left = meta["turn_left"]
        self.turn_right = meta["turn_right"]
        self.jump_nodes = meta["jump_nodes"]


class MaleCNSModel:
    """ornata/fly LIF dynamics with a market stimulus replacing the RGB retina."""

    dt = 0.020
    tau_m = 0.100
    threshold = 1.0

    def __init__(self, cache: Path | None = None, seed: int = 64,
                 connectome: MaleCNSConnectome | None = None):
        if connectome is None:
            if cache is None:
                raise ValueError("cache or connectome is required")
            connectome = MaleCNSConnectome(cache)
        self.connectome = connectome
        self.n = connectome.n
        self.w = connectome.w
        self.visual = connectome.visual
        self.forward = connectome.forward
        self.turn_left = connectome.turn_left
        self.turn_right = connectome.turn_right
        self.jump_nodes = connectome.jump_nodes
        self.rng = np.random.default_rng(seed)
        self.v = np.zeros(self.n, np.float32)
        self.spikes = np.zeros(self.n, np.float32)
        self.history = deque(maxlen=13)
        self.encoder = MarketEncoder(len(self.visual))
        self.current_drive = np.zeros(len(self.visual), dtype=np.float32)
        self.quote_drive = np.zeros(len(self.visual), dtype=np.float32)
        self.current_stimulus: Stimulus | None = None
        self.step_count = 0

    def set_quote(self, quote: MarketSnapshot) -> Stimulus:
        self.quote_drive, self.current_stimulus = self.encoder.encode(quote)
        self.current_drive = self.quote_drive.copy()
        return self.current_stimulus

    def set_stimulus_scale(self, scale: float) -> None:
        self.current_drive = self.quote_drive * float(np.clip(scale, 0.0, 1.0))

    def tick(self) -> NeuralReadout:
        pools = np.concatenate((self.forward, self.turn_left, self.turn_right, self.jump_nodes))
        current = np.asarray(self.w[:, np.flatnonzero(self.spikes)].sum(axis=1)).ravel() * 1.50
        baseline = self.rng.random(self.n) < (1.2 * self.dt)
        self.v *= np.exp(-self.dt / self.tau_m)
        self.v += current + baseline.astype(np.float32) * 0.22 + 0.180
        self.v[self.visual] += self.current_drive * 0.62
        fired = self.v >= self.threshold
        self.v[fired] = 0.0
        self.spikes[:] = fired
        self.history.append(fired[pools].copy())
        self.step_count += 1
        recent = np.stack(tuple(self.history)).mean(axis=0)
        splits = np.cumsum([len(self.forward), len(self.turn_left), len(self.turn_right)])
        forward, left, right, jump = [float(x.mean()) for x in np.split(recent, splits)]
        # Market interpretation: the original forward pool is arousal; original
        # right/left turn pools provide directional buy/sell evidence.
        score = (right - left) * 1100.0
        return NeuralReadout(right, left, max(forward, jump), score, int(fired.sum()))

    def step(self, quote: MarketSnapshot, ticks: int = 13) -> tuple[Stimulus, NeuralReadout]:
        stimulus = self.set_quote(quote)
        if ticks < 1:
            raise ValueError("ticks must be positive")
        activity = self.tick()
        total_spikes = activity.spikes
        for _ in range(ticks - 1):
            activity = self.tick()
            total_spikes += activity.spikes
        return stimulus, NeuralReadout(activity.buy_rate, activity.sell_rate,
                                       activity.arousal_rate, activity.score, total_spikes)

    def save_state(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        history = (np.stack(tuple(self.history)) if self.history else
                   np.empty((0, len(np.concatenate((
                       self.forward, self.turn_left, self.turn_right, self.jump_nodes)))),
                            dtype=np.bool_))
        with temporary.open("wb") as output:
            np.savez_compressed(
                output, v=self.v, spikes=self.spikes, history=history,
                quote_drive=self.quote_drive, current_drive=self.current_drive,
                step_count=np.array([self.step_count], dtype=np.int64),
                rng_state=np.array([json.dumps(self.rng.bit_generator.state)]),
            )
        os.replace(temporary, path)

    def load_state(self, path: Path) -> bool:
        try:
            with np.load(path, allow_pickle=False) as state:
                if state["v"].shape != self.v.shape or state["spikes"].shape != self.spikes.shape:
                    return False
                self.v[:] = state["v"]
                self.spikes[:] = state["spikes"]
                self.quote_drive[:] = state["quote_drive"]
                self.current_drive[:] = state["current_drive"]
                self.history.clear()
                self.history.extend(row.astype(np.bool_) for row in state["history"])
                self.step_count = int(state["step_count"][0])
                self.rng.bit_generator.state = json.loads(str(state["rng_state"][0]))
            return True
        except (FileNotFoundError, KeyError, ValueError, OSError, json.JSONDecodeError):
            return False
