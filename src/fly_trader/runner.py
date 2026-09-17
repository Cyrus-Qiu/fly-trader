from __future__ import annotations

import json
import asyncio
import time
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from .dashboard import DashboardState, start_dashboard
from .accounts import monitor_account, read_longbridge_paper_account
from .market import (AlpacaMarketSource, LongbridgeHKMarketSource,
                     TigerDelayedQuoteSource, TimestampDeduplicator)
from .neural import MaleCNSConnectome, MaleCNSModel
from .pipeline import StrategyPipeline
from .performance import LivePerformanceTracker
from .proposals import OrderProposalEngine
from .runtime import RotatingJsonlWriter, RuntimeHealth, RuntimeStateStore
from .signals import Signal, SignalDecoder
from .experiment import EXPERIMENT_ID


def _restore_runtime(store: RuntimeStateStore, models: dict, pipelines: dict,
                     proposal_engine: OrderProposalEngine) -> bool:
    state = store.load()
    if (state.get("schema") != 2 or state.get("experiment_id") != EXPERIMENT_ID or
            set(state.get("symbols", [])) != set(models)):
        return False
    proposal_engine.restore(state.get("proposal_engine", {}))
    for symbol, pipeline in pipelines.items():
        pipeline.restore(state.get("pipelines", {}).get(symbol, {}))
    state_dir = store.path.with_suffix("")
    restored = [
        model.load_state(state_dir / f"{symbol}.npz")
        for symbol, model in models.items()
    ]
    return bool(restored) and all(restored)


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


def run(symbols: list[str], config: Path, cache: Path, log_path: Path,
        poll_seconds: float = 5.0, once: bool = False, seed: int = 64) -> None:
    source = TigerDelayedQuoteSource(config)
    model = MaleCNSModel(cache, seed=seed)
    decoder = SignalDecoder()
    dedupe = TimestampDeduplicator()
    with RotatingJsonlWriter(log_path) as output:
        while True:
            for quote in source.fetch(symbols):
                if not dedupe.accept(quote):
                    continue
                stimulus, activity = model.step(quote)
                signal = decoder.decode(activity, bool(quote.halted))
                event = {
                    "schema": 1,
                    "mode": "signal-only-no-orders",
                    "quote": quote.public_dict(),
                    "stimulus": asdict(stimulus),
                    "neural_activity": asdict(activity),
                    "signal": asdict(signal),
                }
                output.write(event)
                action_name = {"BUY": "买入", "HOLD": "观望", "SELL": "卖出"}[signal.action]
                print(f"老虎延时行情｜{quote.symbol}｜时间：{quote.market_time_ms}｜"
                      f"信号：{action_name}｜分数：{signal.score:.3f}")
            if once:
                return
            time.sleep(poll_seconds)


async def run_iex(symbols: list[str], cache: Path, log_path: Path,
                  duration: float = 0, seed: int = 64,
                  dashboard_port: int = 8787, open_browser: bool = True,
                  neural_hz: int | None = None,
                  shared_dashboard_state: DashboardState | None = None,
                  shared_connectome: MaleCNSConnectome | None = None) -> None:
    """Run independent 50 Hz neural states sharing one immutable connectome."""
    symbols = list(dict.fromkeys(symbol.upper() for symbol in symbols))
    if not symbols:
        raise ValueError("at least one symbol is required")
    if len(symbols) > 30:
        raise ValueError("Alpaca Basic supports at most 30 streamed symbols")
    sources = [AlpacaMarketSource(symbols, "iex"), AlpacaMarketSource(symbols, "overnight")]
    if shared_connectome is None:
        print(f"正在从 {cache} 加载共享 MaleCNS 连接矩阵……")
    graph = shared_connectome or MaleCNSConnectome(cache)
    neural_hz = neural_hz or (50 if len(symbols) <= 2 else 25)
    models = {symbol: MaleCNSModel(seed=seed + index, connectome=graph)
              for index, symbol in enumerate(symbols)}
    print(f"MaleCNS 已就绪：{graph.n:,} 个神经元，{graph.w.nnz:,} 条连接；"
          f"{len(symbols)} 个独立神经状态，共享一份连接矩阵，目标频率 {neural_hz}Hz")
    dashboard_state = shared_dashboard_state or DashboardState()
    dashboard = (None if shared_dashboard_state is not None else
                 start_dashboard(dashboard_state, dashboard_port, open_browser))
    dedupe = TimestampDeduplicator()
    proposal_engine = OrderProposalEngine()
    pipelines = {symbol: StrategyPipeline(
        neural_hz, proposal_engine=proposal_engine) for symbol in symbols}
    queue: asyncio.Queue = asyncio.Queue(maxsize=4096)
    health = RuntimeHealth()
    state_store = RuntimeStateStore(log_path.with_suffix(".state.json"))
    performance_tracker = LivePerformanceTracker()
    restored = _restore_runtime(state_store, models, pipelines, proposal_engine)
    if restored:
        print("已恢复美股神经状态、信号过滤器和风控冷却状态")
    started = time.monotonic()
    last_checkpoint = started
    workers = min(len(symbols), max(1, os.cpu_count() or 1))
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="malecns")

    async def collect(source: AlpacaMarketSource) -> None:
        async for quote in source.stream():
            accepted = dedupe.accept(quote)
            health.received(accepted)
            if not accepted:
                continue
            if queue.full():
                try:
                    queue.get_nowait()
                    health.dropped_queue_events += 1
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(quote)

    collectors = [asyncio.create_task(collect(source)) for source in sources]
    quotes = {symbol: None for symbol in symbols}
    stimuli = {symbol: None for symbol in symbols}
    last_quote_at = {symbol: None for symbol in symbols}
    raw_signals = {symbol: Signal("HOLD", 0.0, "not started") for symbol in symbols}
    stable_signals = {symbol: Signal("HOLD", 0.0, "not started") for symbol in symbols}
    next_tick = time.monotonic()
    last_wait_notice = started - 10.0
    last_status_print = started
    last_printed_action = {symbol: "HOLD" for symbol in symbols}
    last_logged_tracks = {symbol: None for symbol in symbols}
    interval_events = {symbol: 0 for symbol in symbols}
    counts = {symbol: {"BUY": 0, "HOLD": 0, "SELL": 0} for symbol in symbols}
    first_price = {symbol: None for symbol in symbols}
    last_price = {symbol: None for symbol in symbols}
    with RotatingJsonlWriter(log_path) as output:
        try:
            while duration <= 0 or time.monotonic() - started < duration:
                for collector in collectors:
                    if collector.done():
                        error = collector.exception()
                        if error is not None:
                            raise error
                        raise RuntimeError("Alpaca market-data stream ended unexpectedly")
                updated_symbols: set[str] = set()
                while not queue.empty():
                    quote = queue.get_nowait()
                    symbol = quote.symbol
                    if symbol not in models:
                        continue
                    quotes[symbol] = quote
                    stimuli[symbol] = models[symbol].set_quote(quote)
                    last_quote_at[symbol] = time.monotonic()
                    interval_events[symbol] += 1
                    updated_symbols.add(symbol)
                now = time.monotonic()
                ages = {symbol: (float("inf") if last_quote_at[symbol] is None
                                 else now - last_quote_at[symbol]) for symbol in symbols}
                scales = {symbol: (1.0 if ages[symbol] <= 2.0 else
                                   (5.0 - ages[symbol]) / 3.0 if ages[symbol] < 5.0 else 0.0)
                          for symbol in symbols}
                for symbol in symbols:
                    models[symbol].set_stimulus_scale(scales[symbol])
                if dashboard_state.is_paused():
                    dashboard_state.update({
                        "status": "paused", "symbol": symbols[0],
                    })
                    next_tick = time.monotonic() + 1.0 / neural_hz
                    await asyncio.sleep(1.0 / neural_hz)
                    continue
                tick_started = time.monotonic()
                loop = asyncio.get_running_loop()
                activities = await asyncio.gather(*[
                    loop.run_in_executor(executor, models[symbol].tick) for symbol in symbols
                ])
                tick_latency_ms = (time.monotonic() - tick_started) * 1000.0
                health.tick(tick_latency_ms)
                for symbol, activity in zip(symbols, activities):
                    quote = quotes[symbol]
                    pipeline_result = pipelines[symbol].evaluate(
                        activity, quote, now - started,
                        dashboard_state.get_account("US"), ages[symbol])
                    raw_signal = pipeline_result.raw_signal
                    stable_signal = pipeline_result.filtered_signal
                    raw_signals[symbol] = raw_signal
                    stable_signals[symbol] = stable_signal
                    proposal = pipeline_result.proposal
                    if proposal is not None:
                        dashboard_state.update_proposal(symbol, proposal)
                    source_name = None if quote is None else (
                        "IEX成交" if quote.feed == "iex" else "隔夜指示报价")
                    filter_diagnostics = pipeline_result.filter_diagnostics
                    asset_state = {
                        "status": "running", "elapsed_s": now - started, "symbol": symbol,
                        "market": "US",
                        "source_name": source_name, "price": quote.close if quote else None,
                        "quote_age_s": ages[symbol], "stimulus_scale": scales[symbol],
                        "raw_action": raw_signal.action, "raw_score": raw_signal.score,
                        "stable_action": stable_signal.action, "smooth_score": stable_signal.score,
                        "window_mean_score": filter_diagnostics["window_mean_score"],
                        "enter_threshold": filter_diagnostics["enter_threshold"],
                        "exit_threshold": filter_diagnostics["exit_threshold"],
                        "distance_to_trigger": filter_diagnostics["distance_to_trigger"],
                        "calibration_status": filter_diagnostics["calibration_status"],
                        "warmup_remaining_s": filter_diagnostics["warmup_remaining_s"],
                        "baseline_samples": filter_diagnostics["baseline_samples"],
                        "model_step": models[symbol].step_count, "tick_latency_ms": tick_latency_ms,
                        "runtime_health": health.snapshot(),
                    }
                    dashboard_state.update_asset(symbol, asset_state, selected=symbol == symbols[0],
                                                 append=symbol in updated_symbols)
                    if quote is None or stimuli[symbol] is None:
                        continue
                    if symbol in updated_symbols:
                        first_price[symbol] = (quote.close if first_price[symbol] is None
                                               else first_price[symbol])
                        last_price[symbol] = quote.close
                    signature = (
                        raw_signal.action, stable_signal.action,
                        pipeline_result.executable_signal.action,
                    )
                    track_changed = signature != last_logged_tracks[symbol]
                    if symbol not in updated_symbols and not track_changed:
                        continue
                    counts[symbol][stable_signal.action] += 1
                    if symbol in updated_symbols:
                        event = StrategyPipeline.event(
                            quote=quote, stimulus=stimuli[symbol], activity=activity,
                            result=pipeline_result, neural_hz=neural_hz,
                            model_step=models[symbol].step_count,
                            model_dt_s=models[symbol].dt,
                            tick_latency_ms=tick_latency_ms,
                            quote_age_s=ages[symbol], stimulus_scale=scales[symbol],
                            elapsed_s=now - started,
                            reference_price=first_price[symbol],
                            extra={
                                "shared_connectome": True,
                                "simulation_time_factor": neural_hz * models[symbol].dt,
                                "decision_trigger": "quote",
                            },
                        )
                    else:
                        event = StrategyPipeline.transition_event(
                            quote=quote, activity=activity, result=pipeline_result,
                            neural_hz=neural_hz,
                            model_step=models[symbol].step_count,
                            quote_age_s=ages[symbol],
                            stimulus_scale=scales[symbol],
                        )
                    event["runtime_health"] = health.snapshot()
                    performance = performance_tracker.update(
                        symbol, quote.close, pipeline_result)
                    dashboard_state.update_performance(symbol, performance)
                    if symbol in updated_symbols:
                        event["live_performance"] = performance
                    output.write(event)
                    last_logged_tracks[symbol] = signature
                if not any(quotes.values()) and time.monotonic() - last_wait_notice >= 10.0:
                    print("正在等待 IEX 成交或隔夜指示性报价……")
                    last_wait_notice = time.monotonic()
                for symbol in symbols:
                    stable_signal = stable_signals[symbol]
                    if quotes[symbol] is not None and stable_signal.action != last_printed_action[symbol]:
                        action_name = {"BUY": "买入", "HOLD": "观望", "SELL": "卖出"}[stable_signal.action]
                        previous_name = {"BUY": "买入", "HOLD": "观望", "SELL": "卖出"}[last_printed_action[symbol]]
                        print(f"稳定信号变化｜{symbol}｜{previous_name} → {action_name}｜"
                              f"平滑分数：{stable_signal.score:.3f}")
                        last_printed_action[symbol] = stable_signal.action
                if time.monotonic() - last_status_print >= 5.0:
                    summaries = []
                    for symbol in symbols:
                        quote = quotes[symbol]
                        action = {"BUY": "买", "HOLD": "观", "SELL": "卖"}[stable_signals[symbol].action]
                        price = "—" if quote is None else f"{quote.close:.2f}"
                        summaries.append(f"{symbol} {price}/{action}/{interval_events[symbol]}条")
                        interval_events[symbol] = 0
                    print(f"双标的摘要｜{'｜'.join(summaries)}｜神经批次耗时：{tick_latency_ms:.1f}ms")
                    last_status_print = time.monotonic()
                if time.monotonic() - last_checkpoint >= 30.0:
                    _save_runtime(state_store, "US", models, pipelines,
                                  proposal_engine, health, last_price)
                    last_checkpoint = time.monotonic()
                next_tick += 1.0 / neural_hz
                delay = next_tick - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                else:
                    next_tick = time.monotonic()
                    await asyncio.sleep(0)
        finally:
            for collector in collectors:
                collector.cancel()
            await asyncio.gather(*collectors, return_exceptions=True)
            executor.shutdown(wait=True, cancel_futures=True)
            _save_runtime(state_store, "US", models, pipelines,
                          proposal_engine, health, last_price)
            if dashboard is not None:
                dashboard.shutdown()
                dashboard.server_close()
            total = sum(sum(value.values()) for value in counts.values())
            print(f"运行结束：共记录 {total} 条多标的行情；未启用任何下单功能")


async def run_longbridge(symbols: list[str], cache: Path, log_path: Path,
                         duration: float = 0, seed: int = 64,
                         poll_seconds: float = 5.0, dashboard_port: int = 8787,
                         open_browser: bool = True,
                         shared_dashboard_state: DashboardState | None = None,
                         shared_connectome: MaleCNSConnectome | None = None) -> None:
    """Run the existing neural pipeline from read-only Longbridge HK quotes."""
    symbols = [LongbridgeHKMarketSource._normalize_symbol(symbol) for symbol in symbols]
    source = LongbridgeHKMarketSource(symbols, poll_seconds)
    if shared_connectome is None:
        print(f"正在从 {cache} 加载共享 MaleCNS 连接矩阵……")
    graph = shared_connectome or MaleCNSConnectome(cache)
    models = {symbol: MaleCNSModel(seed=seed + index, connectome=graph)
              for index, symbol in enumerate(symbols)}
    dedupe = TimestampDeduplicator()
    proposal_engine = OrderProposalEngine()
    pipelines = {symbol: StrategyPipeline(
        50, baseline_min_samples=30, proposal_engine=proposal_engine) for symbol in symbols}
    started = time.monotonic()
    dashboard_state = shared_dashboard_state or DashboardState()
    dashboard = (None if shared_dashboard_state is not None else
                 start_dashboard(dashboard_state, dashboard_port, open_browser))
    health = RuntimeHealth()
    state_store = RuntimeStateStore(log_path.with_suffix(".state.json"))
    performance_tracker = LivePerformanceTracker()
    restored = _restore_runtime(state_store, models, pipelines, proposal_engine)
    if restored:
        print("已恢复港股神经状态、信号过滤器和风控冷却状态")
    last_checkpoint = started
    neural_hz = 50
    queue: asyncio.Queue = asyncio.Queue(maxsize=4096)
    executor = ThreadPoolExecutor(
        max_workers=min(len(symbols), max(1, os.cpu_count() or 1)),
        thread_name_prefix="malecns-hk",
    )
    quotes = {symbol: None for symbol in symbols}
    stimuli = {symbol: None for symbol in symbols}
    last_quote_at = {symbol: None for symbol in symbols}
    last_actions = {symbol: "HOLD" for symbol in symbols}
    last_logged_tracks = {symbol: None for symbol in symbols}
    print(f"MaleCNS 已就绪：{len(symbols)} 个港股独立神经状态；"
          f"持续 {neural_hz}Hz，只记录信号，不下单")

    async def collect() -> None:
        async for quote in source.stream():
            accepted = dedupe.accept(quote) and quote.close > 0
            health.received(accepted)
            if not accepted:
                continue
            if queue.full():
                try:
                    queue.get_nowait()
                    health.dropped_queue_events += 1
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(quote)

    collector = asyncio.create_task(collect())
    next_tick = time.monotonic()
    try:
        with RotatingJsonlWriter(log_path) as output:
            while duration <= 0 or time.monotonic() - started < duration:
                if collector.done():
                    error = collector.exception()
                    if error is not None:
                        raise error
                    raise RuntimeError("长桥行情流意外结束")
                updated: set[str] = set()
                while not queue.empty():
                    quote = queue.get_nowait()
                    quotes[quote.symbol] = quote
                    stimuli[quote.symbol] = models[quote.symbol].set_quote(quote)
                    last_quote_at[quote.symbol] = time.monotonic()
                    updated.add(quote.symbol)
                now = time.monotonic()
                ages = {
                    symbol: (float("inf") if last_quote_at[symbol] is None
                             else now - last_quote_at[symbol])
                    for symbol in symbols
                }
                scales = {
                    symbol: (1.0 if ages[symbol] <= 2 else
                             (5 - ages[symbol]) / 3 if ages[symbol] < 5 else 0.0)
                    for symbol in symbols
                }
                for symbol in symbols:
                    models[symbol].set_stimulus_scale(scales[symbol])
                if dashboard_state.is_paused():
                    next_tick = time.monotonic() + 1 / neural_hz
                    await asyncio.sleep(1 / neural_hz)
                    continue
                tick_started = time.monotonic()
                loop = asyncio.get_running_loop()
                activities = await asyncio.gather(*[
                    loop.run_in_executor(executor, models[symbol].tick)
                    for symbol in symbols
                ])
                tick_latency_ms = (time.monotonic() - tick_started) * 1000
                health.tick(tick_latency_ms)
                elapsed = now - started
                for symbol, activity in zip(symbols, activities):
                    quote = quotes[symbol]
                    result = pipelines[symbol].evaluate(
                        activity, quote, elapsed,
                        dashboard_state.get_account("HK"), ages[symbol])
                    if result.proposal is not None:
                        dashboard_state.update_proposal(symbol, result.proposal)
                    diagnostics = result.filter_diagnostics
                    mode = ("等待行情" if quote is None else
                            "实时" if quote.feed == "longbridge-hk-realtime" else "延迟/轮询")
                    dashboard_state.update_asset(symbol, {
                        "status": "running", "elapsed_s": elapsed,
                        "symbol": symbol, "market": "HK",
                        "source_name": f"长桥港股{mode}",
                        "price": quote.close if quote else None,
                        "quote_age_s": ages[symbol],
                        "quote_received_at_ms": (
                            int(time.time() * 1000 - ages[symbol] * 1000)
                            if quote else None),
                        "stimulus_scale": scales[symbol],
                        "raw_action": result.raw_signal.action,
                        "raw_score": result.raw_signal.score,
                        "stable_action": result.filtered_signal.action,
                        "smooth_score": result.filtered_signal.score,
                        "window_mean_score": diagnostics["window_mean_score"],
                        "enter_threshold": diagnostics["enter_threshold"],
                        "exit_threshold": diagnostics["exit_threshold"],
                        "distance_to_trigger": diagnostics["distance_to_trigger"],
                        "calibration_status": diagnostics["calibration_status"],
                        "warmup_remaining_s": diagnostics["warmup_remaining_s"],
                        "baseline_samples": diagnostics["baseline_samples"],
                        "model_step": models[symbol].step_count,
                        "tick_latency_ms": tick_latency_ms,
                        "runtime_health": health.snapshot(),
                    }, selected=symbol == symbols[0], append=symbol in updated)
                    if quote is None or stimuli[symbol] is None:
                        continue
                    signature = (
                        result.raw_signal.action, result.filtered_signal.action,
                        result.executable_signal.action,
                    )
                    track_changed = signature != last_logged_tracks[symbol]
                    if symbol not in updated and not track_changed:
                        continue
                    if symbol in updated:
                        event = StrategyPipeline.event(
                            quote=quote, stimulus=stimuli[symbol], activity=activity,
                            result=result, neural_hz=neural_hz,
                            model_step=models[symbol].step_count,
                            model_dt_s=models[symbol].dt,
                            tick_latency_ms=tick_latency_ms,
                            quote_age_s=ages[symbol],
                            stimulus_scale=scales[symbol], elapsed_s=elapsed,
                            extra={
                                "shared_connectome": True,
                                "decision_trigger": "quote",
                            },
                        )
                    else:
                        event = StrategyPipeline.transition_event(
                            quote=quote, activity=activity, result=result,
                            neural_hz=neural_hz,
                            model_step=models[symbol].step_count,
                            quote_age_s=ages[symbol],
                            stimulus_scale=scales[symbol],
                        )
                    event["runtime_health"] = health.snapshot()
                    performance = performance_tracker.update(
                        symbol, quote.close, result)
                    dashboard_state.update_performance(symbol, performance)
                    if symbol in updated:
                        event["live_performance"] = performance
                    output.write(event)
                    last_logged_tracks[symbol] = signature
                    if result.filtered_signal.action != last_actions[symbol]:
                        action = {"BUY": "买入", "HOLD": "观望",
                                  "SELL": "卖出"}[result.filtered_signal.action]
                        print(f"港股稳定信号变化｜{symbol}｜{action}｜"
                              f"分数：{result.filtered_signal.score:.3f}")
                        last_actions[symbol] = result.filtered_signal.action
                if time.monotonic() - last_checkpoint >= 30.0:
                    _save_runtime(state_store, "HK", models, pipelines,
                                  proposal_engine, health)
                    last_checkpoint = time.monotonic()
                next_tick += 1 / neural_hz
                delay = next_tick - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                else:
                    next_tick = time.monotonic()
                    await asyncio.sleep(0)
    finally:
        collector.cancel()
        await asyncio.gather(collector, return_exceptions=True)
        executor.shutdown(wait=True, cancel_futures=True)
        _save_runtime(state_store, "HK", models, pipelines,
                      proposal_engine, health)
        if dashboard is not None:
            dashboard.shutdown()
            dashboard.server_close()
    print("港股行情运行结束；未启用任何下单功能")


async def run_both(us_symbols: list[str], hk_symbols: list[str], cache: Path,
                   log_path: Path, duration: float = 0, seed: int = 64,
                   poll_seconds: float = 5.0, dashboard_port: int = 8787,
                   open_browser: bool = True, neural_hz: int | None = None) -> None:
    """Run US and HK read-only feeds into one dashboard, never creating trade clients."""
    state = DashboardState()
    print(f"正在从 {cache} 加载双市场共享 MaleCNS 连接矩阵……")
    graph = MaleCNSConnectome(cache)
    print("MaleCNS 连接矩阵加载完成，正在启动仪表盘……")
    dashboard = start_dashboard(state, dashboard_port, open_browser)
    us_log = log_path.with_name(f"{log_path.stem}-us{log_path.suffix}")
    hk_log = log_path.with_name(f"{log_path.stem}-hk{log_path.suffix}")
    print(f"双市场模式｜美股日志：{us_log}｜港股日志：{hk_log}")

    async def guard(market_name: str, job) -> None:
        try:
            await job
        except PermissionError as error:
            print(f"{market_name}行情鉴权失败：{error}；另一市场将继续运行")
        except Exception as error:
            print(f"{market_name}行情服务停止：{type(error).__name__}: {error}；"
                  "另一市场将继续运行")

    try:
        def update_longbridge_account(_market: str, values: dict) -> None:
            # One OpenAPI paper ledger contains both HK and US positions.
            state.update_account("HK", values)
            state.update_account("US", values)

        account_tasks = [asyncio.create_task(monitor_account(
            "长桥", read_longbridge_paper_account, update_longbridge_account))]
        await asyncio.gather(
            guard("美股", run_iex(us_symbols, cache, us_log, duration, seed,
                                  dashboard_port, False, neural_hz, state, graph)),
            guard("港股", run_longbridge(hk_symbols, cache, hk_log, duration,
                                         seed + 1000, poll_seconds, dashboard_port,
                                         False, state, graph)),
        )
    finally:
        for task in locals().get("account_tasks", []):
            task.cancel()
        await asyncio.gather(*locals().get("account_tasks", []), return_exceptions=True)
        dashboard.shutdown()
        dashboard.server_close()
