import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from fly_trader.accounts import INITIAL_CASH, LocalAccount
from fly_trader.dashboard import DashboardState, start_dashboard
from fly_trader.market import MarketSnapshot
from fly_trader.neural import NeuralReadout
from fly_trader.proposals import OrderProposalEngine
from fly_trader.session import ExperimentSession, StateConflict, validate_config
from fly_trader.signals import Signal


def quote(symbol="AAPL", price=100, stamp="2026-09-16T14:00:00+00:00"):
    ms = int(datetime.fromisoformat(stamp).timestamp() * 1000)
    return MarketSnapshot(symbol, ms, stamp, price, price, price, price, price, 10, 0, feed="iex")


def proposal(symbol="AAPL", side="BUY", quantity=10, price=100):
    return dict(status="ready", symbol=symbol, side=side, quantity=quantity, reference_price=price)


def test_local_ledger_costs_shared_cash_no_duplicate_and_sell():
    a, hk = LocalAccount("US"), LocalAccount("HK")
    a.mark(quote())
    a.mark(quote("MSFT"))
    a.execute(proposal(), "one", "now")
    assert a.cash == pytest.approx(INITIAL_CASH - 1000.8)
    assert a.execute(proposal(), "one", "now") is None
    a.execute(proposal("MSFT"), "two", "now")
    assert a.cash == pytest.approx(INITIAL_CASH - 2001.6)
    assert hk.cash == INITIAL_CASH
    a.mark(quote(price=110))
    assert a.snapshot()["unrealized_pl"] == pytest.approx(98.4)
    a.execute(proposal(side="SELL", price=110), "three", "now")
    assert a.realized == pytest.approx(98.32)
    assert "AAPL" not in a.positions
    assert a.execute(proposal(side="SELL"), "four", "now") is None
    assert a.execute(proposal(quantity=1_000_000), "five", "now") is None
    snap = a.snapshot()
    assert snap["equity"] - INITIAL_CASH == pytest.approx(snap["realized_pl"] + snap["unrealized_pl"])


def test_valuation_daily_baseline_and_stale():
    a = LocalAccount("US")
    a.mark(quote())
    a.execute(proposal(), "one", "now")
    a.mark(quote(price=80))
    previous = a.equity
    assert a.max_drawdown < 0
    a.mark(quote(price=90, stamp="2026-09-17T14:00:00+00:00"))
    assert a.day_start_equity == previous
    assert a.snapshot(now_ms=2_000_000_000_000)["positions"][0]["stale"]


def test_dynamic_symbols_lots_and_cost_inclusive_risk():
    a = LocalAccount("US")
    q = quote()
    a.mark(q)
    engine = OrderProposalEngine(allowed_symbols=["AAPL"], confirmation_s=0)
    ready = engine.evaluate("AAPL", Signal("BUY", 30, ""), q, a.snapshot(), now=1)
    assert ready["status"] == "ready"
    poor = {**a.snapshot(), "cash": 100.01}
    engine = OrderProposalEngine(allowed_symbols=["AAPL"], confirmation_s=0)
    assert engine.evaluate("AAPL", Signal("BUY", 30, ""), q, poor, now=1)["status"] == "blocked"
    hk = OrderProposalEngine(allowed_symbols=["9988.HK"], confirmation_s=0)
    result = hk.evaluate("9988.HK", Signal("BUY", 30, ""), quote("9988.HK"), a.snapshot(), now=0)
    assert "每手" in result["reason"]


def test_config_normalization():
    assert validate_config({"us_symbols": " aapl,MSFT aapl", "hk_symbols": "00700，9988.HK"}) == {
        "us_symbols": ["AAPL", "MSFT"], "hk_symbols": ["700.HK", "9988.HK"], "duration_s": 3600}
    assert validate_config({"hk_symbols": "12345", "duration_s": 1})["us_symbols"] == []
    assert validate_config({"us_symbols": ["BRK.B"], "duration_s": 0}, True)["duration_s"] == 0


@pytest.mark.parametrize("payload", [
    {}, {"us_symbols": "../x"}, {"hk_symbols": "AAPL"}, {"hk_symbols": "00000"},
    {"us_symbols": "AAPL", "duration_s": 0}, {"us_symbols": "AAPL", "duration_s": -1},
    {"us_symbols": "AAPL", "duration_s": float("nan")},
    {"us_symbols": "AAPL", "duration_s": float("inf")},
    {"us_symbols": "AAPL", "duration_s": True}, {"us_symbols": [1]},
    {"us_symbols": [f"A{i}" for i in range(31)]},
])
def test_invalid_config(payload):
    with pytest.raises(ValueError):
        validate_config(payload)


def test_deadline_pause_and_second_run_reset(tmp_path, monkeypatch):
    import fly_trader.runner as runner
    async def scenario():
        clock = [10.0]
        state = DashboardState()
        session = ExperimentSession(state, Path("unused"), tmp_path, clock=lambda: clock[0])
        async def prepare():
            return {"US": ()}
        async def market(*args):
            while session.active():
                await asyncio.sleep(.001)
        monkeypatch.setattr(session, "prepare", prepare)
        monkeypatch.setattr(runner, "run_market", market)
        await session.start({"us_symbols": "AAPL", "duration_s": 5})
        await asyncio.sleep(.01)
        first_id = session.run_id
        await session.pause(True)
        with pytest.raises(StateConflict):
            await session.start({"us_symbols": "AAPL"})
        session.accounts["US"].cash = 123
        clock[0] = 16
        assert not session.active()
        await session.task
        report = state.get_report()
        assert session.status == "completed"
        assert report["reason"] == "duration_elapsed"
        assert report["symbols"]["AAPL"]["has_data"] is False
        assert not report["complete"]
        assert json.loads((session.run_dir / "report.json").read_text(encoding="utf-8")) == report
        with pytest.raises(StateConflict):
            await session.pause(False)
        await session.start({"us_symbols": "MSFT", "duration_s": 5})
        assert session.accounts["US"].cash == INITIAL_CASH
        assert not state.snapshot()["assets"]
        assert state.get_report() is None
        assert session.run_id != first_id
        await session.stop()
        await session.task
        assert session.report["reason"] == "user_stopped"
    asyncio.run(scenario())


def test_partial_market_failure(tmp_path, monkeypatch):
    import fly_trader.runner as runner
    async def scenario():
        state = DashboardState()
        session = ExperimentSession(state, Path("unused"), tmp_path)
        async def prepare():
            return {"US": (), "HK": ()}
        async def market(s, market):
            if market == "US":
                raise PermissionError("secret must never appear in report")
            while session.active():
                await asyncio.sleep(.005)
        monkeypatch.setattr(session, "prepare", prepare)
        monkeypatch.setattr(runner, "run_market", market)
        await session.start({"us_symbols": "AAPL", "hk_symbols": "9988", "duration_s": .05})
        await session.task
        assert session.reason == "duration_elapsed"
        assert "US" in session.report["errors"]
        assert "secret" not in json.dumps(session.report)
        assert not session.report["complete"]
    asyncio.run(scenario())


def test_http_lifecycle_and_report_stays_available(tmp_path, monkeypatch):
    import fly_trader.runner as runner
    async def scenario():
        session = ExperimentSession(DashboardState(), Path("unused"), tmp_path)
        async def prepare():
            return {"US": ()}
        async def market(*args):
            while session.active():
                await asyncio.sleep(.005)
        monkeypatch.setattr(session, "prepare", prepare)
        monkeypatch.setattr(runner, "run_market", market)
        server = start_dashboard(session.state, 0, False, controller=session, loop=asyncio.get_running_loop())
        base = f"http://127.0.0.1:{server.server_port}"
        def call(path, payload=None):
            request = Request(base + path, data=None if payload is None else json.dumps(payload).encode(),
                              headers={"Content-Type": "application/json"})
            try:
                with urlopen(request, timeout=3) as response:
                    return response.status, json.load(response)
            except HTTPError as error:
                return error.code, json.load(error)
        try:
            assert (await asyncio.to_thread(call, "/api/experiment/start", {}))[0] == 400
            assert (await asyncio.to_thread(call, "/api/pause", {"paused": True}))[0] == 409
            config = {"us_symbols": "AAPL", "duration_s": 10}
            assert (await asyncio.to_thread(call, "/api/experiment/start", config))[0] == 200
            assert (await asyncio.to_thread(call, "/api/experiment/start", config))[0] == 409
            assert (await asyncio.to_thread(call, "/api/pause", {"paused": True}))[0] == 200
            assert (await asyncio.to_thread(call, "/api/experiment/stop", {}))[0] == 200
            await session.task
            code, report = await asyncio.to_thread(call, "/api/experiment/report")
            assert code == 200 and report == session.report
            code, state = await asyncio.to_thread(call, "/api/state")
            assert code == 200 and state["experiment"]["status"] == "completed"
        finally:
            await session.close()
            await asyncio.to_thread(server.shutdown)
            server.server_close()
    asyncio.run(scenario())


def test_runner_deadline_during_tick_never_executes(tmp_path, monkeypatch):
    import fly_trader.runner as runner
    async def scenario():
        clock = [0.0]
        session = ExperimentSession(DashboardState(), Path("unused"), tmp_path, clock=lambda: clock[0])
        session.config = {"duration_s": 1, "us_symbols": ["AAPL"], "hk_symbols": []}
        session.started = 0
        session.run_id = "test"
        session.run_dir = tmp_path
        class Source:
            async def stream(self):
                yield quote()
                await asyncio.Event().wait()
        class Model:
            step_count = 0
            def set_stimulus_scale(self, scale):
                pass
            def tick(self):
                clock[0] = 2
                return NeuralReadout(.2, 0, .1, 25, 5)
        monkeypatch.setattr(runner, "_save_runtime", lambda *args: None)
        await runner.run_market(session, "US", ["AAPL"], [Source()], {"AAPL": Model()}, {}, 50)
        assert not session.accounts["US"].trades
        assert not session.state.snapshot()["performance"]
    asyncio.run(scenario())

def test_runner_applies_shared_pool_trades_and_persists_events(tmp_path, monkeypatch):
    import fly_trader.runner as runner
    from fly_trader.neural import Stimulus
    from fly_trader.pipeline import StrategyPipeline
    class ImmediatePipeline(StrategyPipeline):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.filter.adaptive = False
            self.filter.enter_threshold = .1
            self.filter.exit_threshold = .04
            self.proposal_engine.confirmation_s = 0
    class Model:
        step_count = 0
        dt = .02
        def tick(self):
            self.step_count += 1
            return NeuralReadout(.2, 0, .1, 25, 5)
        def set_stimulus_scale(self, scale):
            pass
        def set_quote(self, q):
            return Stimulus(0, 0, 0, 0, 0, 0, .5)
    class Source:
        async def stream(self):
            yield quote("AAPL")
            yield quote("MSFT")
            await asyncio.Event().wait()
    async def scenario():
        session = ExperimentSession(DashboardState(), Path("unused"), tmp_path)
        async def prepare():
            return {"US": (["AAPL", "MSFT"], [Source()], {s: Model() for s in ["AAPL", "MSFT"]}, {}, 50)}
        monkeypatch.setattr(session, "prepare", prepare)
        monkeypatch.setattr(runner, "_save_runtime", lambda *args: None)
        monkeypatch.setattr(runner, "StrategyPipeline", ImmediatePipeline)
        monkeypatch.setattr(runner.time, "time", lambda: quote().market_time_ms / 1000)
        await session.start({"us_symbols": ["AAPL", "MSFT"], "duration_s": .15})
        await session.task
        account = session.report["accounts"]["US"]
        assert account["cash"] == pytest.approx(INITIAL_CASH - 200.16)
        assert account["trade_count"] == 2
        assert [t["symbol"] for t in session.report["trades"]["US"]] == ["AAPL", "MSFT"]
        assert len(account["positions"]) == 2
        events = [json.loads(line) for line in (session.run_dir / "signals-us.jsonl").read_text(encoding="utf-8").splitlines()]
        assert all(e["run_id"] == session.run_id for e in events)
        assert sum(e["local_trade"] is not None for e in events) == 2
        fills = [json.loads(line) for line in (session.run_dir / "trades-us.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(fills) == 8
        assert sum(f["track"] == "risk_executable" for f in fills) == 2
        tracks = session.report["track_accounts"]["US"]
        assert tracks["fly_raw"]["bought_quantity"] == 9992
        assert tracks["risk_executable"]["bought_quantity"] == 2
        assert tracks["disconnected_connectome"]["equity"] == INITIAL_CASH
        assert sum(len(v) for v in session.report["track_trades"]["US"].values()) == 8
    asyncio.run(scenario())


def test_all_market_failure_and_preparation_failure(tmp_path, monkeypatch):
    import fly_trader.runner as runner
    async def scenario():
        session = ExperimentSession(DashboardState(), Path("unused"), tmp_path)
        async def prepare():
            return {"US": ()}
        async def market(*args):
            raise RuntimeError("disconnected")
        monkeypatch.setattr(session, "prepare", prepare)
        monkeypatch.setattr(runner, "run_market", market)
        await session.start({"us_symbols": "AAPL", "duration_s": 10})
        await session.task
        assert session.status == "failed"
        assert session.report["reason"] == "market_failed"
        async def broken_prepare():
            raise FileNotFoundError("model cache missing")
        monkeypatch.setattr(session, "prepare", broken_prepare)
        await session.start({"us_symbols": "AAPL", "duration_s": 10})
        await session.task
        assert session.report["reason"] == "preparation_failed"
        assert session.report["started_at"] is None
        assert (session.run_dir / "report.json").exists()
    asyncio.run(scenario())

def test_hk_lot_metadata_from_quote_api(monkeypatch):
    import longbridge.openapi as sdk
    from fly_trader.market import LongbridgeHKMarketSource
    for key in ("LONGBRIDGE_APP_KEY", "LONGBRIDGE_APP_SECRET", "LONGBRIDGE_ACCESS_TOKEN"):
        monkeypatch.setenv(key, "test-placeholder")
    class QuoteContext:
        @staticmethod
        def create(config):
            return QuoteContext()
        async def quote(self, symbols):
            return []
        async def static_info(self, symbols):
            assert symbols == ["9988.HK", "700.HK"]
            return [SimpleNamespace(symbol="09988.HK", lot_size=100),
                    SimpleNamespace(symbol="00700.HK", lot_size=0)]
    monkeypatch.setattr(sdk, "AsyncQuoteContext", QuoteContext)
    result = asyncio.run(LongbridgeHKMarketSource(["9988.HK", "700.HK"]).lot_sizes())
    assert result == {"9988.HK": 100}


def test_track_funds_independent_and_equal_symbol_budgets():
    from fly_trader.performance import TrackPerformance
    from fly_trader.pipeline import PipelineResult
    pools = TrackPerformance("US", ["AAPL", "MSFT"])
    for symbol in pools.symbols:
        pools.mark(quote(symbol))
    result = PipelineResult(Signal("BUY", 10, ""), Signal("HOLD", 0, ""), None, {})
    fills = pools.update("AAPL", 100, result, "one", "now", {})
    assert {f["track"] for f in fills} == {"fly_raw", "buy_and_hold"}
    snap = pools.snapshot()
    assert snap["fly_raw"]["bought_quantity"] == 4996
    assert snap["fly_raw"]["cash"] == pytest.approx(500000.32)
    assert snap["fly_filtered"]["cash"] == INITIAL_CASH
    assert snap["risk_executable"]["cash"] == INITIAL_CASH
    assert snap["disconnected_connectome"]["cash"] == INITIAL_CASH
    assert pools.update("AAPL", 100, result, "repeat", "now", {}) == []
    pools.update("MSFT", 100, result, "two", "now", {})
    assert pools.snapshot()["fly_raw"]["cash"] == pytest.approx(.64)
    assert TrackPerformance("HK", ["9988.HK"]).snapshot()["fly_raw"]["cash"] == INITIAL_CASH


def test_closed_position_statistics_and_reinvestment():
    from fly_trader.performance import TrackPerformance
    from fly_trader.pipeline import PipelineResult
    pools = TrackPerformance("US", ["AAPL"])
    buy = PipelineResult(Signal("BUY", 10, ""), Signal("HOLD", 0, ""), None, {})
    sell = PipelineResult(Signal("SELL", -10, ""), Signal("HOLD", 0, ""), None, {})
    pools.mark(quote())
    pools.update("AAPL", 100, buy, "one", "now", {})
    pools.mark(quote(price=110))
    pools.update("AAPL", 110, sell, "two", "now", {})
    account = pools.snapshot()["fly_raw"]
    record = account["symbol_records"][0]
    assert record["bought_quantity"] == record["sold_quantity"] == 9992
    assert record["quantity"] == 0
    assert record["cost_price"] is None
    assert record["average_buy_price"] == 100
    assert record["average_sell_price"] == 110
    assert record["unrealized_pl"] == 0
    assert record["realized_pl"] == pytest.approx(9992 * (110 * .9992 - 100 * 1.0008))
    assert account["return"] == pytest.approx(record["realized_pl"] / INITIAL_CASH)
    assert pools.snapshot()["buy_and_hold"]["sold_quantity"] == 0
    pools.update("AAPL", 110, buy, "three", "now", {})
    record = pools.snapshot()["fly_raw"]["symbol_records"][0]
    assert record["bought_quantity"] > 9992
    assert record["cost_price"] == pytest.approx(110 * 1.0008)
    assert pools.snapshot()["fly_raw"]["cash"] >= 0


def test_hk_track_lots_and_missing_lots_block_execution():
    from fly_trader.performance import TrackPerformance
    from fly_trader.pipeline import PipelineResult
    pools = TrackPerformance("HK", ["9988.HK"])
    pools.mark(quote("9988.HK"))
    buy = PipelineResult(Signal("BUY", 10, ""), Signal("BUY", 10, ""), None, {})
    assert pools.update("9988.HK", 100, buy, "one", "now", {}) == []
    assert "每手" in pools.snapshot()["fly_raw"]["symbol_records"][0]["execution_note"]
    fills = pools.update("9988.HK", 100, buy, "two", "now", {"9988.HK": 100})
    assert all(f["quantity"] == 9900 for f in fills)
    assert len(fills) == 3
    assert pools.snapshot()["fly_filtered"]["cash"] == pytest.approx(9208)


def test_calibration_distinguishes_market_time_and_sample_waits():
    from fly_trader.signals import NeuralSignalFilter
    f = NeuralSignalFilter(adaptive=True, calibration_s=300, baseline_min_samples=3)
    buy = Signal("BUY", 30, "background")
    f.update(buy, 90, stale=True)
    d = f.diagnostics()
    assert d["calibration_status"] == "waiting_market"
    assert d["warmup_remaining_s"] == 210
    assert d["baseline_samples"] == 0
    f.update(buy, 310, stale=True)
    assert f.diagnostics()["calibration_status"] == "waiting_market"
    f.update(buy, 311)
    assert f.diagnostics()["calibration_status"] == "waiting_samples"
    assert f.diagnostics()["baseline_min_samples"] == 3
    f.update(buy, 312)
    f.update(buy, 313)
    assert f.diagnostics()["calibration_status"] == "ready"
    f.update(buy, 314, stale=True)
    assert f.diagnostics()["calibration_status"] == "ready"
    assert f.diagnostics()["market_stale"]


def test_calibration_diagnostics_use_same_elapsed_time_as_decision():
    from fly_trader.signals import NeuralSignalFilter
    f = NeuralSignalFilter(adaptive=True, calibration_s=10, baseline_min_samples=1)
    f.update(Signal("BUY", 30, ""), 5)
    f.update(Signal("BUY", 30, ""), 11, stale=True)
    assert f.diagnostics()["calibration_status"] == "ready"
    restored = NeuralSignalFilter(adaptive=True, calibration_s=10, baseline_min_samples=1)
    restored.restore(f.snapshot())
    assert restored.diagnostics()["calibration_status"] == "ready"


def test_market_coverage_hint_uses_eastern_time_and_dst():
    from fly_trader.calendar import TradingCalendar
    calendar = TradingCalendar()
    summer = quote(stamp="2026-09-17T08:30:00+00:00")
    winter = quote(stamp="2026-12-17T09:30:00+00:00")
    for q in (summer, winter):
        hint = calendar.market_data_hint("TSLA", q.market_time_ms)
        assert "04:30" in hint
        assert "盘前" in hint
    evening = quote(stamp="2026-09-17T22:00:00+00:00")
    assert "盘后" in calendar.market_data_hint("TSLA", evening.market_time_ms)


def test_no_quotes_never_trades_on_neural_background_and_explains_why(tmp_path, monkeypatch):
    import fly_trader.runner as runner
    class SilentSource:
        async def stream(self):
            await asyncio.Event().wait()
            yield
    class Model:
        step_count = 0
        dt = .02
        def tick(self):
            self.step_count += 1
            return NeuralReadout(.2, 0, .1, 42, 5)
        def set_stimulus_scale(self, scale):
            pass
    async def scenario():
        session = ExperimentSession(DashboardState(), Path("unused"), tmp_path)
        async def prepare():
            return {"US": (["TSLA"], [SilentSource()], {"TSLA": Model()}, {}, 50)}
        monkeypatch.setattr(session, "prepare", prepare)
        monkeypatch.setattr(runner, "_save_runtime", lambda *args: None)
        await session.start({"us_symbols": "TSLA", "duration_s": .08})
        await session.task
        asset = session.state.snapshot()["assets"]["TSLA"]
        assert asset["raw_action"] == "BUY"
        assert asset["price"] is None
        assert asset["calibration_status"] == "waiting_market"
        assert asset["warmup_remaining_s"] > 299
        for track, account in session.report["track_accounts"]["US"].items():
            assert account["bought_quantity"] == 0
            if track != "disconnected_connectome":
                assert "缺少成交价格" in account["symbol_records"][0]["execution_note"]
    asyncio.run(scenario())
