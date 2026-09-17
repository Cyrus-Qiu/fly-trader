from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import time
from uuid import uuid4

from .accounts import SIMULATION_VERSION
from .performance import TrackPerformance
from .experiment import experiment_identity


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def normalize_symbols(value, market):
    if isinstance(value, str):
        value = re.split(r"[\s,，]+", value.strip())
    if not isinstance(value, list) or any(not isinstance(s, str) for s in value):
        raise ValueError("股票代码必须是文本或文本列表")
    symbols = []
    for raw in value:
        symbol = raw.strip().upper()
        if not symbol:
            continue
        if market == "HK":
            if not re.fullmatch(r"[0-9]{1,5}(?:\.HK)?", symbol):
                raise ValueError(f"港股代码无效：{symbol}")
            number = int(symbol.removesuffix(".HK"))
            if number == 0:
                raise ValueError("港股代码不能为 0")
            symbol = f"{number}.HK"
        elif not re.fullmatch(r"[A-Z][A-Z0-9]*(?:[.-][A-Z0-9]+)?", symbol) or len(symbol) > 20 or symbol.endswith(".HK"):
            raise ValueError(f"美股代码无效：{symbol}")
        if symbol not in symbols:
            symbols.append(symbol)
    if market == "US" and len(symbols) > 30:
        raise ValueError("美股最多订阅 30 个标的")
    return symbols


def validate_config(payload, allow_unlimited=False):
    if not isinstance(payload, dict):
        raise ValueError("配置必须为 JSON 对象")
    us = normalize_symbols(payload.get("us_symbols", []), "US")
    hk = normalize_symbols(payload.get("hk_symbols", []), "HK")
    if not us and not hk:
        raise ValueError("至少填写一只港股或美股")
    raw = payload.get("duration_s", 3600)
    if isinstance(raw, bool):
        raise ValueError("实验时长必须为正数")
    try:
        duration = float(raw)
    except (ValueError, TypeError):
        raise ValueError("实验时长必须为正数") from None
    if not math.isfinite(duration) or duration < 0 or (duration == 0 and not allow_unlimited):
        raise ValueError("实验时长必须为有限正数")
    return {"us_symbols": us, "hk_symbols": hk, "duration_s": duration}


class StateConflict(Exception):
    pass


class ExperimentSession:
    """All mutations run on the asyncio loop; HTTP reads published locked snapshots."""
    def __init__(self, state, cache: Path, log_root: Path, seed=64, poll_seconds=5,
                 neural_hz=None, clock=time.monotonic):
        self.state, self.cache, self.log_root = state, cache, log_root
        self.seed, self.poll_seconds, self.neural_hz = seed, poll_seconds, neural_hz
        self.clock = clock
        self.status = "idle"
        self.task = None
        self.report = None
        self.config = {}
        self.run_id = None
        self.started = None
        self.started_at = None
        self.ended_at = None
        self.elapsed = 0.0
        self.stop_requested = False
        self.reason = None
        self.errors = {}
        self.warnings = []
        self.market_settings = {}
        self.portfolios = {m: TrackPerformance(m, self.config.get(key, []))
                           for m, key in (("US", "us_symbols"), ("HK", "hk_symbols"))}
        self.accounts = {m: p.accounts["risk_executable"] for m, p in self.portfolios.items()}
        self.publish()
        self.publish_accounts()

    def publish_accounts(self):
        for market, account in self.accounts.items():
            self.state.update_account(market, account.snapshot())
            self.state.update_track_accounts(market, self.portfolios[market].snapshot())

    def publish(self):
        duration = self.config.get("duration_s", 0)
        elapsed = self.elapsed if self.started is None or self.status in {"completed", "failed"} else self.clock() - self.started
        self.state.update_experiment({
            "status": self.status, "run_id": self.run_id, "config": self.config,
            "started_at": self.started_at, "ended_at": self.ended_at,
            "elapsed_s": elapsed,
            "remaining_s": max(0, duration - elapsed) if duration else None,
            "reason": self.reason, "errors": dict(self.errors), "warnings": list(self.warnings),
            "report_available": self.report is not None,
        })

    async def start(self, payload, allow_unlimited=False):
        if self.task is not None and not self.task.done():
            raise StateConflict("实验正在进行，请先结束当前实验")
        config = validate_config(payload, allow_unlimited)
        self.config = config
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:10]
        self.run_dir = self.log_root / self.run_id
        self.status, self.report = "preparing", None
        self.started = self.started_at = self.ended_at = None
        self.elapsed = 0.0
        self.stop_requested, self.reason = False, None
        self.errors, self.warnings = {}, []
        self.market_settings = {}
        self.portfolios = {m: TrackPerformance(m, self.config.get(key, []))
                           for m, key in (("US", "us_symbols"), ("HK", "hk_symbols"))}
        self.accounts = {m: p.accounts["risk_executable"] for m, p in self.portfolios.items()}
        self.state.reset_run()
        self.publish_accounts()
        self.publish()
        self.task = asyncio.create_task(self._run())
        return {"run_id": self.run_id, "status": self.status}

    async def stop(self):
        if self.status not in {"preparing", "running", "paused"}:
            raise StateConflict("当前没有可以结束的实验")
        self.stop_requested, self.reason, self.status = True, "user_stopped", "stopping"
        self.publish()
        return {"status": self.status}

    async def pause(self, paused):
        if not isinstance(paused, bool):
            raise ValueError("paused 必须为布尔值")
        if self.status not in {"running", "paused"} or not self.active():
            raise StateConflict("仅运行中的实验可以暂停或继续")
        self.state.set_paused(paused)
        self.status = "paused" if paused else "running"
        self.publish()
        return {"paused": paused}

    def active(self):
        return (not self.stop_requested and self.started is not None
                and (self.config["duration_s"] == 0 or
                     self.clock() - self.started < self.config["duration_s"]))

    async def prepare(self):
        from .market import LongbridgeUSMarketSource, LongbridgeHKMarketSource
        from .neural import MaleCNSConnectome, MaleCNSModel
        graph = await asyncio.to_thread(MaleCNSConnectome, self.cache)
        prepared = {}
        for market, key in (("US", "us_symbols"), ("HK", "hk_symbols")):
            symbols = self.config[key]
            if not symbols or self.stop_requested:
                continue
            try:
                sources = ([LongbridgeUSMarketSource(symbols, self.poll_seconds)]
                           if market == "US" else [LongbridgeHKMarketSource(symbols, self.poll_seconds)])
                try:
                    await sources[0].initialize()
                except ConnectionError:
                    self.warnings.append(f"{market} 长桥暂未连接，运行期间将继续重试")
                lots = {}
                if market == "HK":
                    try:
                        lots = await asyncio.wait_for(sources[0].lot_sizes(), 15)
                    except Exception:
                        self.warnings.append("港股每手资料读取失败，相关标的仅生成信号，不模拟成交")
                    missing = [s for s in symbols if not lots.get(s)]
                    if missing:
                        self.warnings.append("缺少每手股数：" + ", ".join(missing))
                models = {}
                for index, symbol in enumerate(symbols):
                    if self.stop_requested:
                        break
                    models[symbol] = await asyncio.to_thread(
                        MaleCNSModel, seed=self.seed + index + (1000 if market == "HK" else 0),
                        connectome=graph)
                if not self.stop_requested:
                    hz = 50 if market == "HK" else (self.neural_hz or (50 if len(symbols) <= 2 else 25))
                    self.market_settings[market] = {
                        "provider": "longbridge", "enable_overnight": market == "US",
                        "access_point": sources[0].endpoint,
                        "neural_hz": hz, "seed": self.seed + (1000 if market == "HK" else 0),
                        "lot_sizes": {s: lots.get(s) if market == "HK" else 1 for s in symbols},
                    }
                    prepared[market] = (symbols, sources, models, lots, hz)
            except Exception as error:
                self.errors[market] = f"{type(error).__name__}：行情或模型准备失败"
        return prepared

    async def _run(self):
        from .runner import run_market
        tasks = []
        try:
            self.run_dir.mkdir(parents=True, exist_ok=False)
            prepared = await self.prepare()
            if not self.stop_requested and prepared:
                self.started, self.started_at = self.clock(), utc_now()
                self.status = "running"
                self.publish()
                async def guarded(market, values):
                    try:
                        await run_market(self, market, *values)
                    except Exception as error:
                        self.errors[market] = f"{type(error).__name__}：行情服务停止"
                tasks = [asyncio.create_task(guarded(market, values)) for market, values in prepared.items()]
                while self.active() and not all(t.done() for t in tasks):
                    self.publish()
                    await asyncio.sleep(0.1)
                if self.reason is None:
                    self.reason = "duration_elapsed" if not self.active() else "market_failed"
            elif not self.stop_requested:
                self.reason = "preparation_failed"
        except asyncio.CancelledError:
            self.reason = "process_stopped"
        except Exception as error:
            self.errors["experiment"] = f"{type(error).__name__}：实验准备或运行失败"
            self.reason = "preparation_failed"
        finally:
            self.stop_requested, self.status = True, "stopping"
            self.elapsed = 0.0 if self.started is None else self.clock() - self.started
            self.ended_at = utc_now()
            self.publish()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            self.state.set_paused(False)
            self.publish_accounts()
            self.status = "failed" if self.reason in {"preparation_failed", "market_failed"} else "completed"
            self.report = self.build_report()
            try:
                self.run_dir.mkdir(parents=True, exist_ok=True)
                target = self.run_dir / "report.json"
                temporary = target.with_suffix(".tmp")
                temporary.write_text(json.dumps(self.report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
                temporary.replace(target)
            except OSError:
                self.status = "failed"
                self.errors["report"] = "报告保存失败；可从页面下载内存中的结果"
                self.report["errors"] = dict(self.errors)
                self.report["complete"] = False
            self.state.set_report(self.report)
            self.publish()

    def build_report(self):
        snapshot = self.state.snapshot()
        diagnostics = {}
        for symbol in self.config["us_symbols"] + self.config["hk_symbols"]:
            asset = snapshot["assets"].get(symbol, {})
            diagnostics[symbol] = {
                "has_data": asset.get("price") is not None,
                "warmup_complete": asset.get("calibration_status") == "ready",
                "last_price": asset.get("price"),
                "valuation_time_ms": asset.get("market_time_ms"),
                "stale": asset.get("market_time_ms") is None or
                    datetime.now(timezone.utc).timestamp() * 1000 - asset["market_time_ms"] > 5000,
            }
        return {
            "run_id": self.run_id, "simulation_version": SIMULATION_VERSION,
            "experiment": experiment_identity(), "config": self.config,
            "market_settings": self.market_settings,
            "started_at": self.started_at, "ended_at": self.ended_at,
            "elapsed_s": self.elapsed, "reason": self.reason,
            "complete": not self.errors and not self.warnings and all(d["has_data"] for d in diagnostics.values()),
            "errors": dict(self.errors), "warnings": list(self.warnings),
            "accounts": snapshot["accounts"], "symbols": diagnostics,
            "track_accounts": snapshot["track_accounts"],
            "track_trades": {m: {t: a.trades for t, a in p.accounts.items()}
                             for m, p in self.portfolios.items()},
            "trades": {m: a.trades for m, a in self.accounts.items()},
            "signal_comparisons": snapshot["track_accounts"],
            "accounts_scope": "risk_executable_compatibility_alias",
            "comparison_scope": "independent_track_market_accounts",
            "fee_bps": 3, "slippage_bps": 5, "settlement": "last_valid_quote_no_liquidation",
        }

    async def close(self):
        if self.task and not self.task.done():
            self.stop_requested, self.reason = True, "process_stopped"
            await self.task
