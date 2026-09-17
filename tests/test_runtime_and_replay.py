import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from fly_trader.calendar import TradingCalendar
from fly_trader.replay import evaluate
from fly_trader.runtime import RotatingJsonlWriter, RuntimeStateStore
from fly_trader.experiment import public_experiment_spec


def test_calendar_blocks_exchange_holiday_and_out_of_coverage():
    calendar = TradingCalendar()
    def timestamp(value: str) -> int:
        local = datetime.fromisoformat(value).replace(
            tzinfo=ZoneInfo("America/New_York"))
        return int(local.timestamp() * 1000)
    assert "休市日" in calendar.session_reason("NVDA", timestamp("2026-12-25T10:00:00"))
    assert "提前收市" in calendar.session_reason("NVDA", timestamp("2026-12-24T14:00:00"))
    assert "超出" in calendar.session_reason("NVDA", timestamp("2028-01-03T10:00:00"))


def test_rotating_jsonl_and_atomic_state(tmp_path: Path):
    log = tmp_path / "events.jsonl"
    log.write_text('{"complete":true}\n{"partial":', encoding="utf-8")
    with RotatingJsonlWriter(log, max_bytes=1024, backups=2) as writer:
        for index in range(40):
            writer.write({"index": index, "payload": "x" * 80})
    assert log.exists()
    assert log.with_suffix(".jsonl.1").exists()
    for path in (log, log.with_suffix(".jsonl.1")):
        for line in path.read_text(encoding="utf-8").splitlines():
            json.loads(line)
    assert "partial" not in log.with_suffix(".jsonl.1").read_text(encoding="utf-8")

    store = RuntimeStateStore(tmp_path / "runtime.json")
    store.save({"schema": 1, "step": 42})
    assert store.load()["step"] == 42
    (tmp_path / "runtime.json").write_text("{broken", encoding="utf-8")
    assert store.load() == {}


def test_replay_evaluates_strategy_and_three_baselines(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    actions = ["BUY", "HOLD", "SELL", "HOLD"]
    prices = [100, 110, 120, 115]
    lines = []
    for index, (action, price) in enumerate(zip(actions, prices)):
        lines.append(json.dumps({
            "schema": 8,
            "experiment": public_experiment_spec(),
            "symbol": "NVDA",
            "quote": {"symbol": "NVDA", "market_time_ms": 1000 + index * 1000,
                      "close": price},
            "filtered_signal": {"action": action, "score": 0, "reason": "test"},
            "tracks": {
                "fly_raw": {"action": action},
                "fly_filtered": {"action": action},
                "risk_executable": {"action": "HOLD"},
            },
        }))
    path.write_text("\n".join(lines) + "\n{invalid\n", encoding="utf-8")
    report = evaluate([path], fee_bps=0, slippage_bps=0, seed=7)
    nvda = report["symbols"]["NVDA"]
    assert nvda["fly_raw"]["total_return"] == pytest.approx(.2)
    assert nvda["fly_filtered"]["total_return"] == pytest.approx(.2)
    assert nvda["risk_executable"]["total_return"] == 0
    assert nvda["buy_and_hold"]["total_return"] == pytest.approx(.15)
    assert nvda["disconnected_connectome"]["total_return"] == 0
    assert "random_same_frequency" in nvda
    assert report["diagnostics"]["invalid_lines"] == 1


def test_replay_excludes_legacy_events_by_default(tmp_path: Path):
    path = tmp_path / "legacy.jsonl"
    path.write_text(json.dumps({
        "schema": 7, "symbol": "NVDA",
        "quote": {"symbol": "NVDA", "market_time_ms": 1, "close": 100},
        "raw_signal": {"action": "BUY"},
        "filtered_signal": {"action": "BUY"},
    }) + "\n", encoding="utf-8")
    report = evaluate([path])
    assert report["symbols"] == {}
    assert report["diagnostics"]["legacy_events_skipped"] == 1
    legacy = evaluate([path], include_legacy=True)
    assert legacy["symbols"]["NVDA"]["events"] == 1
