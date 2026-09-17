from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from .runner import run, run_both, run_iex, run_longbridge
from .replay import evaluate, write_report


def main() -> None:
    # The project-local secret file is the selected credential source. This also
    # prevents stale values left in a PowerShell session from shadowing edits.
    load_dotenv(Path(".env"), override=True)
    parser = argparse.ArgumentParser(description="MaleCNS market signal experiment; never submits orders")
    parser.add_argument("--source", choices=("both", "iex", "tiger", "longbridge", "replay"),
                        default="both",
                        help="iex starts both IEX trades and free overnight indicative quotes")
    parser.add_argument("--symbols", nargs="+",
                        default=["NVDA"])
    parser.add_argument("--hk-symbols", nargs="+", default=["700.HK", "2513.HK"],
                        help="Hong Kong symbols used by --source both")
    parser.add_argument("--config", type=Path, default=Path("config/tiger_openapi_config.properties"))
    parser.add_argument("--cache", type=Path, default=Path("data/malecns"))
    parser.add_argument("--log", type=Path, default=Path("logs/signals.jsonl"))
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=64)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--duration", type=float, default=0,
                        help="Alpaca stream run seconds; zero runs until Ctrl+C")
    parser.add_argument("--dashboard-port", type=int, default=8787)
    parser.add_argument("--no-browser", action="store_true",
                        help="start dashboard without opening the browser")
    parser.add_argument("--neural-hz", type=int,
                        help="override neural rate; defaults to 25Hz for 3+ symbols, otherwise 50Hz")
    parser.add_argument("--replay-log", type=Path, nargs="+",
                        help="JSONL event files used by --source replay")
    parser.add_argument("--report", type=Path,
                        help="optional JSON output path for replay evaluation")
    parser.add_argument("--fee-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--include-legacy", action="store_true",
                        help="include pre-schema-8 events in replay; excluded by default")
    args = parser.parse_args()
    if args.source == "replay":
        if not args.replay_log:
            parser.error("--source replay requires --replay-log")
        report = evaluate(args.replay_log, args.fee_bps, args.slippage_bps,
                          args.seed, args.include_legacy)
        print(write_report(report, args.report))
    elif args.source == "both":
        try:
            asyncio.run(run_both(args.symbols, args.hk_symbols, args.cache, args.log,
                                 args.duration, args.seed, args.poll_seconds,
                                 args.dashboard_port, not args.no_browser, args.neural_hz))
        except KeyboardInterrupt:
            pass
        except RuntimeError as error:
            parser.error(str(error))
    elif args.source == "iex":
        try:
            asyncio.run(run_iex(args.symbols, args.cache, args.log, args.duration, args.seed,
                                args.dashboard_port, not args.no_browser, args.neural_hz))
        except KeyboardInterrupt:
            pass
    elif args.source == "longbridge":
        try:
            asyncio.run(run_longbridge(args.symbols, args.cache, args.log, args.duration,
                                       args.seed, args.poll_seconds, args.dashboard_port,
                                       not args.no_browser))
        except KeyboardInterrupt:
            pass
        except RuntimeError as error:
            parser.error(str(error))
    else:
        run(args.symbols, args.config, args.cache, args.log, args.poll_seconds, args.once, args.seed)


if __name__ == "__main__":
    main()
