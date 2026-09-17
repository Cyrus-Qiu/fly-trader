from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from fly_trader.market import (IEXTradeAccumulator, LongbridgeHKMarketSource, MarketSnapshot,
                               OvernightQuoteAccumulator, TimestampDeduplicator,
                               _is_authentication_failure, _longbridge_snapshot,
                               snapshots_from_frame)
from fly_trader.neural import (MaleCNSConnectome, MaleCNSModel, MarketEncoder,
                               NeuralReadout, Stimulus)
from fly_trader.pipeline import PipelineResult, StrategyPipeline
from fly_trader.performance import LivePerformanceTracker
from fly_trader.signals import NeuralSignalFilter, Signal, SignalDecoder
from fly_trader.proposals import OrderProposalEngine
import fly_trader.accounts as accounts


def quote(timestamp=1000, close=102.0, halted=0):
    return MarketSnapshot("AAPL", timestamp, "2026-01-01T00:00:00+00:00",
                          100, 101, 103, 99, close, 1000, halted)


def market_ms(value: str, zone: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=ZoneInfo(zone)).timestamp() * 1000)


def test_timestamp_deduplication_rejects_duplicates_and_old_data():
    dedupe = TimestampDeduplicator()
    assert dedupe.accept(quote(1000))
    assert not dedupe.accept(quote(1000))
    assert not dedupe.accept(quote(999))
    assert dedupe.accept(quote(1001))


def test_tiger_frame_conversion_uses_provider_time():
    frame = pd.DataFrame([dict(symbol="aapl", pre_close=100, time=1234, volume=5,
                               open=100, high=102, low=99, close=101, halted=0)])
    result = snapshots_from_frame(frame)[0]
    assert result.symbol == "AAPL"
    assert result.market_time_ms == 1234


def test_market_encoder_direction_and_halt():
    encoder = MarketEncoder(60)
    up, up_meta = encoder.encode(quote(close=102))
    halted, halted_meta = encoder.encode(quote(timestamp=1001, halted=3))
    assert up_meta.return_from_previous_close > 0
    assert up[:10].mean() > up[10:20].mean()
    assert halted_meta.halted == 1
    assert halted.sum() == 0


def test_iex_trade_accumulator_builds_ohlcv_and_uses_exchange_time():
    accumulator = IEXTradeAccumulator()
    first = accumulator.ingest({"T": "t", "S": "AAPL", "p": 100.0, "s": 2,
                                "t": "2026-09-16T13:30:00.123456789Z"})
    second = accumulator.ingest({"T": "t", "S": "AAPL", "p": 101.0, "s": 3,
                                 "t": "2026-09-16T13:30:00.223456789Z"})
    assert first is not None and second is not None
    assert first.market_time_ms == 1789565400123
    assert second.pre_close == 100.0
    assert second.open == 100.0
    assert second.high == 101.0
    assert second.low == 100.0
    assert second.close == 101.0
    assert second.volume == 5
    assert second.feed == "iex"
    assert second.event_type == "trade"


def test_overnight_quote_uses_midpoint_and_rejects_crossed_market():
    accumulator = OvernightQuoteAccumulator()
    first = accumulator.ingest({"T": "q", "S": "AAPL", "bp": 99.9, "ap": 100.1,
                                "t": "2026-09-16T02:00:00.123456789Z"})
    second = accumulator.ingest({"T": "q", "S": "AAPL", "bp": 101.8, "ap": 102.0,
                                 "t": "2026-09-16T02:00:01.123456789Z"})
    crossed = accumulator.ingest({"T": "q", "S": "AAPL", "bp": 103.0, "ap": 102.0,
                                  "t": "2026-09-16T02:00:02Z"})
    assert first is not None and second is not None
    assert first.close == 100.0
    assert second.close == 101.9
    assert second.high == 101.9
    assert second.volume == 0
    assert second.feed == "overnight"
    assert second.event_type == "indicative_quote"
    assert crossed is None


def test_overnight_quote_rejects_wide_spread_and_large_jump():
    accumulator = OvernightQuoteAccumulator(max_spread_bps=100, max_jump_fraction=.02)
    assert accumulator.ingest({"T": "q", "S": "AAPL", "bp": 99.9, "ap": 100.1,
                               "t": "2026-09-16T02:00:00Z"}) is not None
    assert accumulator.ingest({"T": "q", "S": "AAPL", "bp": 98, "ap": 102,
                               "t": "2026-09-16T02:00:01Z"}) is None
    assert accumulator.ingest({"T": "q", "S": "AAPL", "bp": 102.9, "ap": 103.1,
                               "t": "2026-09-16T02:00:02Z"}) is None


def test_neural_signal_filter_smooths_and_uses_hysteresis():
    filter_ = NeuralSignalFilter(hz=10, window_s=.2, ema_half_life_s=.01,
                                 enter_threshold=20, exit_threshold=8, cooldown_s=0)
    buy = Signal("BUY", 42, "raw")
    neutral = Signal("HOLD", 0, "raw")
    assert filter_.update(buy, 0.0).action == "BUY"
    # One neutral sample leaves the two-sample window mean above the exit gate.
    assert filter_.update(neutral, .1).action == "BUY"
    assert filter_.update(neutral, .2).action == "HOLD"
    diagnostics = filter_.diagnostics()
    assert diagnostics["samples"] == 2
    assert diagnostics["enter_threshold"] == 20


def test_neural_signal_filter_forces_hold_when_quote_is_stale():
    filter_ = NeuralSignalFilter(hz=10, window_s=.1, ema_half_life_s=.01,
                                 enter_threshold=20, exit_threshold=8, cooldown_s=0)
    buy = Signal("BUY", 42, "raw")
    assert filter_.update(buy, 0.0).action == "BUY"
    assert filter_.update(buy, .1, stale=True).action == "HOLD"


def test_adaptive_filter_warms_up_then_uses_its_own_percentile_threshold():
    filter_ = NeuralSignalFilter(
        hz=10, window_s=.1, ema_half_life_s=.01, cooldown_s=0,
        adaptive=True, calibration_s=1.0, baseline_min_samples=3,
        enter_percentile=.95, min_enter_threshold=2, max_enter_threshold=20,
    )
    buy = Signal("BUY", 10, "raw")
    assert filter_.update(buy, 0.0).action == "HOLD"
    assert filter_.update(buy, .5).action == "HOLD"
    assert filter_.update(buy, 1.0).action == "BUY"
    diagnostics = filter_.diagnostics()
    assert diagnostics["calibration_status"] == "ready"
    assert diagnostics["baseline_samples"] == 3
    assert 2 <= diagnostics["enter_threshold"] <= 20
    assert diagnostics["distance_to_trigger"] == 0


def test_small_cache_runs_deterministically(tmp_path: Path):
    n = 32
    # visual -> relay -> right-turn creates positive directional evidence.
    rows = np.array([20, 25])
    cols = np.array([0, 20])
    matrix = sparse.csr_matrix((np.array([1.0, 1.0], np.float32), (rows, cols)), shape=(n, n))
    sparse.save_npz(tmp_path / "weights.npz", matrix)
    np.savez(tmp_path / "model.npz", n=n, visual=np.arange(12), forward=np.array([24]),
             turn_left=np.array([26]), turn_right=np.array([25]), jump_nodes=np.array([27]))
    (tmp_path / "manifest.json").write_text("{}")
    a, b = MaleCNSModel(tmp_path, seed=7), MaleCNSModel(tmp_path, seed=7)
    sa, ra = a.step(quote(), ticks=20)
    sb, rb = b.step(quote(), ticks=20)
    assert sa == sb
    assert ra == rb


def test_models_share_one_connectome_matrix(tmp_path: Path):
    n = 32
    sparse.save_npz(tmp_path / "weights.npz", sparse.eye(n, dtype=np.float32))
    np.savez(tmp_path / "model.npz", n=n, visual=np.arange(12), forward=np.array([24]),
             turn_left=np.array([26]), turn_right=np.array([25]), jump_nodes=np.array([27]))
    (tmp_path / "manifest.json").write_text("{}")
    graph = MaleCNSConnectome(tmp_path)
    a = MaleCNSModel(seed=1, connectome=graph)
    b = MaleCNSModel(seed=2, connectome=graph)
    assert a.w is b.w
    assert a.v is not b.v


def test_model_runtime_state_round_trip(tmp_path: Path):
    n = 32
    sparse.save_npz(tmp_path / "weights.npz", sparse.eye(n, dtype=np.float32))
    np.savez(tmp_path / "model.npz", n=n, visual=np.arange(12), forward=np.array([24]),
             turn_left=np.array([26]), turn_right=np.array([25]), jump_nodes=np.array([27]))
    (tmp_path / "manifest.json").write_text("{}")
    original = MaleCNSModel(tmp_path, seed=9)
    original.step(quote(), ticks=3)
    state_path = tmp_path / "state" / "AAPL.npz"
    original.save_state(state_path)
    restored = MaleCNSModel(tmp_path, seed=99)
    assert restored.load_state(state_path)
    assert restored.step_count == original.step_count
    np.testing.assert_array_equal(restored.v, original.v)
    np.testing.assert_array_equal(restored.spikes, original.spikes)
    assert restored.tick() == original.tick()


def test_decoder_never_exposes_an_order_api():
    decoder = SignalDecoder(threshold=8)
    assert not hasattr(decoder, "place_order")


def test_longbridge_hk_symbol_normalization():
    assert LongbridgeHKMarketSource._normalize_symbol("700") == "700.HK"
    assert LongbridgeHKMarketSource._normalize_symbol("00700.hk") == "700.HK"
    assert LongbridgeHKMarketSource._normalize_symbol("02513.hk") == "2513.HK"


def test_longbridge_quote_conversion_preserves_provider_time():
    class Quote:
        timestamp = pd.Timestamp("2026-09-16T01:30:00.123Z").to_pydatetime()
        prev_close, open, high, low, last_done = 100, 101, 103, 99, 102
        volume, trade_status = 1234, "Normal"

    result = _longbridge_snapshot("700.HK", Quote(), "longbridge-hk-bmp", "snapshot")
    assert result.market_time_ms == 1789522200123
    assert result.close == 102
    assert result.feed == "longbridge-hk-bmp"


def test_account_module_exposes_no_order_operation():
    forbidden = {"place_order", "submit_order", "replace_order", "cancel_order"}
    assert forbidden.isdisjoint(dir(accounts))


def test_longbridge_socket_token_url_is_not_treated_as_auth_failure():
    message = "error sending request for url (https://openapi.longbridge.com/v1/socket/token): client error (Connect)"
    assert not _is_authentication_failure(RuntimeError(message))
    assert _is_authentication_failure(RuntimeError("Alpaca stream error 402: auth failed"))


def test_order_proposal_is_allowlisted_confirmed_and_never_submits():
    engine = OrderProposalEngine(confirmation_s=3.0)
    nvda = MarketSnapshot("NVDA", market_ms("2026-09-16T10:00:00", "America/New_York"),
                          "2026-09-16T14:00:00+00:00",
                          200, 200, 201, 199, 200, 10, 0, feed="iex")
    account = {"status": "ok", "equity": 1_000_000, "cash": 1_000_000,
               "fx_rates": {"USD": 7.0, "HKD": 1.0}, "positions": []}
    buy = Signal("BUY", 25, "stable")
    assert engine.evaluate("NVDA", buy, nvda, account, now=0)["status"] == "confirming"
    ready = engine.evaluate("NVDA", buy, nvda, account, now=3)
    assert ready["status"] == "ready"
    assert ready["quantity"] == 1
    assert ready["order_submission_enabled"] is False
    assert not hasattr(engine, "place_order")


def test_order_proposal_allows_zhipu_and_blocks_non_allowlisted_and_naked_sell():
    engine = OrderProposalEngine(confirmation_s=0)
    account = {"status": "ok", "equity": 1_000_000, "cash": 1_000_000,
               "fx_rates": {"USD": 7.0, "HKD": 1.0}, "positions": []}
    aapl = quote(close=100)
    assert engine.evaluate("AAPL", Signal("BUY", 30, "x"), aapl, account,
                           now=0)["status"] == "blocked"
    zhipu = MarketSnapshot("2513.HK", market_ms("2026-09-16T10:00:00", "Asia/Hong_Kong"),
                           "2026-09-16T02:00:00+00:00",
                           500, 500, 505, 495, 500, 10, 0, feed="longbridge-hk")
    assert engine.evaluate("2513.HK", Signal("BUY", 30, "x"), zhipu, account,
                           now=0)["status"] == "ready"
    assert engine.evaluate("2513.HK", Signal("BUY", 30, "x"), zhipu, account,
                           now=0)["quantity"] == 100
    nvda = MarketSnapshot("NVDA", market_ms("2026-09-16T10:00:00", "America/New_York"),
                          "2026-09-16T14:00:00+00:00",
                          200, 200, 201, 199, 200, 10, 0, feed="iex")
    result = engine.evaluate("NVDA", Signal("SELL", -30, "x"), nvda, account, now=0)
    assert result["status"] == "blocked"
    assert "裸卖空" in result["reason"]


def test_order_proposal_blocks_outside_regular_session_and_overnight_quotes():
    engine = OrderProposalEngine(confirmation_s=0)
    account = {"status": "ok", "equity": 100_000, "cash": 100_000,
               "fx_rates": {"USD": 1.0}, "positions": []}
    after_hours = MarketSnapshot(
        "NVDA", market_ms("2026-09-16T18:00:00", "America/New_York"),
        "2026-09-16T22:00:00+00:00", 200, 200, 201, 199, 200, 10, 0, feed="iex")
    result = engine.evaluate("NVDA", Signal("BUY", 30, "x"), after_hours, account, now=0)
    assert result["status"] == "blocked"
    assert "常规交易时段" in result["reason"]

    overnight = MarketSnapshot(
        "NVDA", market_ms("2026-09-17T10:00:00", "America/New_York"),
        "2026-09-17T14:00:00+00:00", 200, 200, 201, 199, 200, 0, 0,
        feed="overnight", event_type="indicative_quote")
    result = engine.evaluate("NVDA", Signal("BUY", 30, "x"), overnight, account, now=1)
    assert result["status"] == "blocked"
    assert "指示性报价" in result["reason"]


def test_order_proposal_enforces_daily_loss_and_five_minute_cooldown():
    engine = OrderProposalEngine(confirmation_s=0, proposal_cooldown_s=300)
    nvda = MarketSnapshot(
        "NVDA", market_ms("2026-09-16T10:00:00", "America/New_York"),
        "2026-09-16T14:00:00+00:00", 200, 200, 201, 199, 200, 10, 0, feed="iex")
    account = {"provider": "paper-test", "status": "ok", "equity": 100_000,
               "day_start_equity": 100_000, "cash": 100_000,
               "fx_rates": {"USD": 1.0}, "positions": []}
    signal = Signal("BUY", 30, "x")
    first = engine.evaluate("NVDA", signal, nvda, account, now=0)
    assert first["status"] == "ready"
    assert all(check["status"] != "pending" for check in first["risk_checks"])
    cooling = engine.evaluate("NVDA", signal, nvda, account, now=60)
    assert cooling["status"] == "blocked"
    assert cooling["cooldown_remaining_s"] == 240
    assert engine.evaluate("NVDA", signal, nvda, account, now=300)["status"] == "ready"

    loss_engine = OrderProposalEngine(confirmation_s=0)
    loss_account = {**account, "equity": 97_999}
    blocked = loss_engine.evaluate("NVDA", signal, nvda, loss_account, now=0)
    assert blocked["status"] == "blocked"
    assert "触发熔断" in blocked["reason"]


def test_strategy_pipeline_builds_shared_versioned_event():
    proposal_engine = OrderProposalEngine(confirmation_s=0, proposal_cooldown_s=0)
    pipeline = StrategyPipeline(50, proposal_engine=proposal_engine)
    pipeline.filter.adaptive = False
    pipeline.filter.enter_threshold = .1
    pipeline.filter.exit_threshold = .04
    nvda = MarketSnapshot(
        "NVDA", market_ms("2026-09-16T10:00:00", "America/New_York"),
        "2026-09-16T14:00:00+00:00", 200, 200, 201, 199, 200, 10, 0, feed="iex")
    account = {"status": "ok", "equity": 100_000, "day_start_equity": 100_000,
               "cash": 100_000, "fx_rates": {"USD": 1.0}, "positions": []}
    activity = NeuralReadout(.2, 0, .1, 25, 5)
    result = pipeline.evaluate(activity, nvda, 10.0, account)
    stimulus = Stimulus(0, 0, 0, 0, 0, 0, .5)
    event = pipeline.event(
        quote=nvda, stimulus=stimulus, activity=activity, result=result,
        neural_hz=50, model_step=10, model_dt_s=.02, elapsed_s=10,
        reference_price=200,
    )
    assert result.filtered_signal.action == "BUY"
    assert result.proposal["status"] == "ready"
    assert event["schema"] == 8
    assert event["symbol"] == "NVDA"
    assert event["experiment"]["primary_track"] == "fly_raw"
    assert event["tracks"]["fly_raw"]["action"] == "BUY"
    assert event["tracks"]["fly_filtered"]["action"] == "BUY"
    assert event["tracks"]["risk_executable"]["action"] == "BUY"
    assert event["order_proposal"]["order_submission_enabled"] is False
    assert event["market_freshness"]["forced_hold"] is False
    transition = pipeline.transition_event(
        quote=nvda, activity=activity, result=result, neural_hz=50,
        model_step=11, quote_age_s=.2, stimulus_scale=1)
    assert transition["schema"] == 8
    assert transition["decision_trigger"] == "neural_transition"
    assert set(transition["quote"]) == {"symbol", "market_time_ms", "close"}
    assert "stimulus" not in transition


def test_live_performance_keeps_raw_filtered_and_risk_tracks_separate():
    tracker = LivePerformanceTracker(fee_bps=0, slippage_bps=0)
    buy = PipelineResult(
        Signal("BUY", 10, "raw"), Signal("HOLD", 0, "filtered"),
        None, {})
    first = tracker.update("NVDA", 100, buy)
    assert first["tracks"]["fly_raw"]["holding"] is True
    assert first["tracks"]["fly_filtered"]["holding"] is False
    assert first["tracks"]["buy_and_hold"]["holding"] is True
    sell = PipelineResult(
        Signal("SELL", -10, "raw"), Signal("HOLD", 0, "filtered"),
        None, {})
    second = tracker.update("NVDA", 110, sell)
    assert second["tracks"]["fly_raw"]["return"] == pytest.approx(.1)
    assert second["tracks"]["fly_raw"]["round_trips"] == 1
    assert second["tracks"]["fly_filtered"]["return"] == 0
