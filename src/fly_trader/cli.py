from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from .runner import serve_experiments
from .session import validate_config
from .replay import evaluate, write_report


def main() -> None:
    # The project-local secret file is the selected credential source. This also
    # prevents stale values left in a PowerShell session from shadowing edits.
    load_dotenv(Path(".env"), override=True)
    parser = argparse.ArgumentParser(description="MaleCNS market signal experiment; never submits orders")
    parser.add_argument("--source", choices=("both", "iex", "longbridge", "replay"),
                        default="both",
                        help="market selection; all live markets use Longbridge (iex is a legacy US-only alias)")
    parser.add_argument("--symbols", nargs="+",
                        default=[])
    parser.add_argument("--hk-symbols", nargs="+", default=[],
                        help="Hong Kong symbols used by --source both")
    parser.add_argument("--cache", type=Path, default=Path("data/malecns"))
    parser.add_argument("--log", type=Path, default=Path("logs/signals.jsonl"))
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=64)
    parser.add_argument("--duration", type=float, default=3600,
                        help="experiment duration in seconds; explicit zero runs until stopped")
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
    parser.add_argument("--auto-start", action="store_true",
                        help="start immediately using CLI symbols and duration")
    args = parser.parse_args()
    if args.source == "replay":
        if not args.replay_log:
            parser.error("--source replay requires --replay-log")
        report = evaluate(args.replay_log, args.fee_bps, args.slippage_bps,
                          args.seed, args.include_legacy)
        print(write_report(report, args.report))
    else:
        config = None
        if args.neural_hz is not None and args.neural_hz <= 0:
            parser.error("--neural-hz must be positive")
        if args.auto_start:
            try:
                config = validate_config({
                    "us_symbols": args.symbols if args.source in {"both", "iex"} else [],
                    "hk_symbols": args.hk_symbols if args.source == "both" else (
                        args.symbols if args.source == "longbridge" else []),
                    "duration_s": args.duration,
                }, allow_unlimited=True)
            except ValueError as error:
                parser.error(str(error))
        try:
            asyncio.run(serve_experiments(
                args.cache, args.log.parent / "runs", args.dashboard_port,
                not args.no_browser, args.seed, args.poll_seconds, args.neural_hz, config))
        except KeyboardInterrupt:
            pass
        except RuntimeError as error:
            parser.error(str(error))


if __name__ == "__main__":
    main()
