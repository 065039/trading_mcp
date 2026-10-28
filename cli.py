from __future__ import annotations

import argparse
import json

from agent_workflow import run_portfolio_cycle


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-agent trading cycle for a watchlist.")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA"], help="Comma-separated or space-separated symbols")
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--timeframe", default="1Day")
    args = parser.parse_args()

    watchlist = []
    for item in args.symbols:
        watchlist.extend([symbol.strip().upper() for symbol in item.split(",") if symbol.strip()])

    report = run_portfolio_cycle(
        watchlist,
        lookback_days=args.lookback_days,
        timeframe=args.timeframe,
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
