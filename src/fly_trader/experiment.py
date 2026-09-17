from __future__ import annotations

import hashlib
import json


EXPERIMENT_ID = "fly-core-v1"

# This specification is intentionally immutable for the lifetime of the
# experiment ID. Any behavioral change must create a new ID instead of editing
# results in place.
EXPERIMENT_SPEC = {
    "experiment_id": EXPERIMENT_ID,
    "primary_track": "fly_raw",
    "interface": {
        "input": "six fixed contiguous visual-receptor bands",
        "return_scale": "2 percent log return maps to +/-1",
        "output": "right-turn rate minus left-turn rate, multiplied by 1100",
        "raw_decoder": "BUY >= 8; SELL <= -8; otherwise HOLD",
    },
    "neural_dynamics": {
        "dt_s": 0.020,
        "tau_m_s": 0.100,
        "threshold": 1.0,
        "synaptic_gain": 1.5,
        "constant_current": 0.180,
        "background_hz": 1.2,
        "background_amplitude": 0.22,
        "visual_gain": 0.62,
    },
    "tracks": {
        "fly_raw": "fixed sensory-motor interface only; primary scientific result",
        "fly_filtered": "EMA, adaptive threshold, hysteresis and warmup comparison",
        "risk_executable": "filtered signal after read-only market/account risk checks",
    },
}

EXPERIMENT_FINGERPRINT = hashlib.sha256(
    json.dumps(EXPERIMENT_SPEC, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def public_experiment_spec() -> dict:
    return {**EXPERIMENT_SPEC, "fingerprint": EXPERIMENT_FINGERPRINT}


def experiment_identity() -> dict:
    return {
        "experiment_id": EXPERIMENT_ID,
        "fingerprint": EXPERIMENT_FINGERPRINT,
        "primary_track": "fly_raw",
    }
