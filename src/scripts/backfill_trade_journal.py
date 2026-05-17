"""Backfill the trade journal from historical backtest CSVs.

Idempotent — re-running won't duplicate rows. Uses (symbol, entry_time) as the
dedup key.

Usage:
    python -m src.scripts.backfill_trade_journal
    python -m src.scripts.backfill_trade_journal reports/relaxed_909_backtest.csv
"""
from __future__ import annotations

from datetime import datetime, time
from pathlib import Path

import polars as pl

from src.risk.trade_journal import append_trade, read_trades


def backfill_from_csv(csv_path: Path) -> int:
    """Append one synthetic journal row per CSV row that is not already present.

    Returns the number of rows actually inserted (excludes duplicates).
    """
    csv_path = Path(csv_path)
    df = pl.read_csv(str(csv_path))

    existing = read_trades()
    existing_keys: set[tuple[str, str]] = set()
    if not existing.is_empty():
        existing_keys = {
            (s, t.isoformat())
            for s, t in zip(
                existing["symbol"].to_list(),
                existing["entry_time"].to_list(),
            )
        }

    inserted = 0
    for row in df.iter_rows(named=True):
        symbol = row["symbol"]
        # Synthesize a midday (11:00 ET) entry so the row lands in the
        # correct YYYY-MM-DD partition for the trading day.
        entry_dt = datetime.combine(
            datetime.strptime(row["date"], "%Y-%m-%d").date(),
            time(11, 0),
        )
        key = (symbol, entry_dt.isoformat())
        if key in existing_keys:
            continue

        pnl = float(row["pnl"])
        # Synthetic record — many fields are unknown for historical CSVs, so
        # we fill conservatively. r_multiple defaults to ±1.0 (1R win/loss).
        record = {
            "symbol": symbol,
            "entry_time": entry_dt,
            "exit_time": entry_dt,
            "entry_price": 0.0,
            "exit_price": 0.0,
            "shares": 0,
            "side": "short",
            "pnl": pnl,
            "r_multiple": 1.0 if pnl > 0 else -1.0,
            "hold_seconds": 0,
            "exit_reason": "backfill",
            "win": pnl > 0,
            "feat_vwap_extension": float(row.get("gain_pct", 0.0)) / 100.0,
            "feat_volume_ratio": 0.0,
            "feat_atr_pct": 0.0,
            "feat_time_of_day_min": 90.0,
            "feat_day_of_week": float(entry_dt.weekday()),
            "feat_factors_count": 0.0,
        }
        append_trade(record)
        existing_keys.add(key)
        inserted += 1
    return inserted


def main() -> None:
    import sys

    csv_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("reports/relaxed_909_backtest.csv")
    )
    n = backfill_from_csv(csv_path)
    print(f"Backfilled {n} rows from {csv_path}")


if __name__ == "__main__":
    main()
