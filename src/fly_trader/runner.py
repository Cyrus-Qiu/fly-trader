from __future__ import annotations

import asyncio
import time
import os
from concurrent.futures import ThreadPoolExecutor

from .dashboard import DashboardState, start_dashboard
from .accounts import SIMULATION_VERSION
from .market import TimestampDeduplicator
from .pipeline import StrategyPipeline
from .proposals import OrderProposalEngine
from .runtime import RotatingJsonlWriter, RuntimeHealth, RuntimeStateStore
from .experiment import EXPERIMENT_ID


def _save_runtime(store: RuntimeStateStore, market: str, models: dict,
                  pipelines: dict, proposal_engine: OrderProposalEngine,
                  health: RuntimeHealth, last_prices: dict | None = None) -> None:
    state_dir = store.path.with_suffix("")
    for symbol, model in models.items():
        model.save_state(state_dir / f"{symbol}.npz")
    store.save({
        "schema": 2,
        "experiment_id": EXPERIMENT_ID,
        "market": market,
        "symbols": list(models),
        "health": health.snapshot(),
        "last_prices": last_prices or {},
        "model_steps": {symbol: model.step_count for symbol, model in models.items()},
        "pipelines": {symbol: pipeline.snapshot()
                      for symbol, pipeline in pipelines.items()},
        "proposal_engine": proposal_engine.snapshot(),
    })


async def run_market(session, market, symbols, sources, models, lots, hz):
    """One sequential ledger per market, sharing the session's deadline."""
    import math
    from .session import utc_now
    state = session.state
    account = session.accounts[market]
    engine = OrderProposalEngine(allowed_symbols=symbols, lot_sizes=lots,
                                 friction=account.friction)
    pipelines = {s: StrategyPipeline(hz, baseline_min_samples=30 if market == "HK" else 100,
                                    proposal_engine=engine) for s in symbols}
    performance = session.portfolios[market]
    health = RuntimeHealth()
    dedupe = TimestampDeduplicator()
    queue = asyncio.Queue(maxsize=4096)
    quotes, stimuli, received = {}, {}, {}
    first_prices = {}
    signatures = {}
    executor = ThreadPoolExecutor(max_workers=min(len(symbols), max(1, os.cpu_count() or 1)),
                                  thread_name_prefix=f"malecns-{market}")

    async def collect(source):
        async for quote in source.stream():
            if not session.active():
                return
            valid = (quote.symbol in models and math.isfinite(quote.close) and quote.close > 0)
            accepted = valid and dedupe.accept(quote)
            health.received(accepted)
            if not accepted:
                continue
            if queue.full():
                queue.get_nowait()
                health.dropped_queue_events += 1
            queue.put_nowait((quote, session.clock()))

    collectors = [asyncio.create_task(collect(source)) for source in sources]
    store = RuntimeStateStore(session.run_dir / f"signals-{market.lower()}.state.json")
    last_checkpoint = session.clock()
    try:
        with RotatingJsonlWriter(session.run_dir / f"signals-{market.lower()}.jsonl") as output, \
                RotatingJsonlWriter(session.run_dir / f"trades-{market.lower()}.jsonl") as trades:
            while session.active():
                for collector in collectors:
                    if collector.done():
                        error = collector.exception()
                        if error:
                            raise error
                        raise RuntimeError("行情流意外结束")
                tick_start = session.clock()
                updated = set()
                while not queue.empty() and session.active():
                    quote, received_at = queue.get_nowait()
                    symbol = quote.symbol
                    quotes[symbol], received[symbol] = quote, received_at
                    first_prices.setdefault(symbol, quote.close)
                    stimuli[symbol] = models[symbol].set_quote(quote)
                    performance.mark(quote)
                    state.update_asset(symbol, {"price": quote.close, "market": market,
                                                "market_time_ms": quote.market_time_ms})
                    updated.add(symbol)
                state.update_account(market, account.snapshot())
                state.update_track_accounts(market, performance.snapshot())
                if not session.active():
                    break
                if state.is_paused():
                    await asyncio.sleep(1 / hz)
                    continue
                now = session.clock()
                # Preserve the frozen signal experiment's receive-age stimulus decay.
                # Provider timestamps are retained separately for valuation warnings.
                ages = {s: max(now - received[s],
                               time.time() - quotes[s].market_time_ms / 1000
                               if quotes[s].feed.startswith("longbridge-") else 0, 0)
                        if s in quotes else math.inf for s in symbols}
                scales = {s: 1.0 if ages[s] <= 2 else max(0, (5 - ages[s]) / 3) for s in symbols}
                for s in symbols:
                    models[s].set_stimulus_scale(scales[s])
                activities = await asyncio.gather(*[
                    asyncio.get_running_loop().run_in_executor(executor, models[s].tick) for s in symbols])
                tick_latency_ms = (session.clock() - now) * 1000
                health.tick(tick_latency_ms)
                for symbol, activity in zip(symbols, activities):
                    if not session.active() or state.is_paused():
                        break
                    quote = quotes.get(symbol)
                    market_reason = (engine.calendar.market_data_hint(symbol, int(time.time() * 1000))
                                     if quote is None else
                                     "行情超过 5 秒未更新，人工过滤与风控轨道暂停成交" if ages[symbol] >= 5 else "")
                    elapsed = session.clock() - session.started
                    result = pipelines[symbol].evaluate(activity, quote, elapsed,
                                                        account.snapshot(), ages[symbol])
                    proposal = result.proposal
                    fills = []
                    if quote is None:
                        performance.wait_for_market(symbol, "未收到有效行情，缺少成交价格")
                    can_trade = (quote is not None and ages[symbol] < 5 and not quote.halted
                                 and (not quote.feed.startswith("longbridge-") or
                                      engine.calendar.simulation_session_reason(symbol, int(time.time() * 1000)) is None)
                                 and (not quote or not quote.feed.startswith("longbridge-us-") or
                                      quote.trade_session == engine.calendar.us_session(int(time.time() * 1000))))
                    if quote is not None and not can_trade:
                        market_reason = ("报价已过期，等待当前时段的新价格" if ages[symbol] >= 5 else
                                         "当前市场或报价时段不允许模拟成交")
                        performance.wait_for_market(symbol, market_reason)
                    if can_trade and session.active():
                        decision_id = f"{session.run_id}:{symbol}:{models[symbol].step_count}"
                        fills = performance.update(symbol, quote.close, result, decision_id, utc_now(), lots)
                        for fill in fills:
                            trades.write({"run_id": session.run_id, "simulation_version": SIMULATION_VERSION, **fill})
                    trade = next((f for f in fills if f["track"] == "risk_executable"), None)
                    if proposal:
                        state.update_proposal(symbol, proposal)
                    state.update_account(market, account.snapshot())
                    state.update_track_accounts(market, performance.snapshot())
                    diagnostics = result.filter_diagnostics
                    asset = {
                        "symbol": symbol, "market": market, "status": "running",
                        "elapsed_s": elapsed, "price": quote.close if quote else None,
                        "market_time_ms": quote.market_time_ms if quote else None,
                        "quote_received_at_ms": int(time.time() * 1000 - ages[symbol] * 1000) if quote else None,
                        "quote_age_s": ages[symbol], "stimulus_scale": scales[symbol],
                        "source_name": quote.feed if quote else "长桥 · 等待行情",
                        "trade_session": quote.trade_session if quote else None,
                        "source_status": [{"status": getattr(source, "status", "unknown"),
                                           "reason": getattr(source, "status_reason", ""),
                                           "access_point": getattr(source, "endpoint", None)} for source in sources],
                        "market_data_status": "waiting" if quote is None else "stale" if not can_trade else "live",
                        "market_data_reason": market_reason,
                        "raw_action": result.raw_signal.action, "raw_score": result.raw_signal.score,
                        "stable_action": result.filtered_signal.action, "smooth_score": result.filtered_signal.score,
                        "model_step": models[symbol].step_count, "tick_latency_ms": tick_latency_ms,
                        "runtime_health": health.snapshot(),
                        **{k: diagnostics[k] for k in (
                            "window_mean_score", "enter_threshold", "exit_threshold", "distance_to_trigger",
                            "calibration_status", "warmup_remaining_s", "baseline_samples",
                            "baseline_min_samples", "market_stale", "calibration_time_complete")},
                    }
                    state.update_asset(symbol, asset, selected=symbol == symbols[0], append=symbol in updated)
                    if quote is None:
                        continue
                    signature = (result.raw_signal.action, result.filtered_signal.action,
                                 result.executable_signal.action)
                    if symbol not in updated and signature == signatures.get(symbol) and not fills:
                        continue
                    common = dict(quote=quote, activity=activity, result=result, neural_hz=hz,
                                  model_step=models[symbol].step_count, quote_age_s=ages[symbol],
                                  stimulus_scale=scales[symbol])
                    if symbol in updated:
                        event = StrategyPipeline.event(**common, stimulus=stimuli[symbol],
                                                       model_dt_s=models[symbol].dt, elapsed_s=elapsed,
                                                       reference_price=first_prices[symbol],
                                                       tick_latency_ms=tick_latency_ms,
                                                       extra={"shared_connectome": True,
                                                              "simulation_time_factor": hz * models[symbol].dt,
                                                              "decision_trigger": "quote"})
                    else:
                        event = StrategyPipeline.transition_event(**common)
                    event.update(run_id=session.run_id, simulation_version=SIMULATION_VERSION,
                                 runtime_health=health.snapshot(), local_trade=trade, local_trades=fills)
                    event["track_accounts"] = performance.snapshot()
                    output.write(event)
                    signatures[symbol] = signature
                if session.clock() - last_checkpoint >= 30 and session.active():
                    await asyncio.to_thread(_save_runtime, store, market, models, pipelines, engine, health)
                    last_checkpoint = session.clock()
                await asyncio.sleep(max(0, 1 / hz - (session.clock() - tick_start)))
    finally:
        for collector in collectors:
            collector.cancel()
        await asyncio.gather(*collectors, return_exceptions=True)
        executor.shutdown(wait=True, cancel_futures=True)
        await asyncio.to_thread(_save_runtime, store, market, models, pipelines, engine, health)


async def serve_experiments(cache, log_root, dashboard_port=8787, open_browser=True,
                            seed=64, poll_seconds=5, neural_hz=None, auto_config=None):
    from .session import ExperimentSession
    state = DashboardState()
    session = ExperimentSession(state, cache, log_root, seed, poll_seconds, neural_hz)
    dashboard = start_dashboard(state, dashboard_port, open_browser,
                                controller=session, loop=asyncio.get_running_loop())
    try:
        if auto_config is not None:
            await session.start(auto_config, allow_unlimited=True)
        await asyncio.Event().wait()
    finally:
        await session.close()
        dashboard.shutdown()
        dashboard.server_close()
