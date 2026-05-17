"""CLI: populate the regime side-table from yfinance.

Usage:
    python -m src.scripts.backfill_regime
    python -m src.scripts.backfill_regime --start 2018-01-01 --end 2024-12-31

Writes data/regime/regime_history.parquet (overwriting any existing file).
Override the destination directory with the REGIME_DIR env var.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta

from src.risk.regime import fetch_regime_history, write_regime_history


def backfill(start: date, end: date) -> int:
    """Fetch the regime classification for [start, end] and persist it.

    Returns the number of rows written (0 if yfinance returned nothing).
    """
    df = fetch_regime_history(start, end)
    if df.is_empty():
        return 0
    write_regime_history(df)
    return df.shape[0]


def main() -> None:
    p = argparse.ArgumentParser(description="Backfill the daily regime side-table.")
    p.add_argument(
        "--start",
        type=lambda s: date.fromisoformat(s),
        default=date.today() - timedelta(days=365 * 6),
        help="Start date (YYYY-MM-DD). Default: ~6 years ago.",
    )
    p.add_argument(
        "--end",
        type=lambda s: date.fromisoformat(s),
        default=date.today(),
        help="End date (YYYY-MM-DD). Default: today.",
    )
    args = p.parse_args()
    n = backfill(args.start, args.end)
    print(f"Regime backfill complete: {n} rows written ({args.start} -> {args.end})")


if __name__ == "__main__":
    main()
