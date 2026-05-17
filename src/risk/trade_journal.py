"""Persistent trade ledger sharded as one parquet per ET trading day."""
import os
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import polars as pl

# Per-process lock; per-day file lock is provided implicitly by atomic
# read-merge-rewrite (single-writer assumed within a process).
_LOCK = threading.Lock()


TRADE_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.Utf8,
    "entry_time": pl.Datetime,
    "exit_time": pl.Datetime,
    "entry_price": pl.Float64,
    "exit_price": pl.Float64,
    "shares": pl.Int64,
    "side": pl.Utf8,
    "pnl": pl.Float64,
    "r_multiple": pl.Float64,
    "hold_seconds": pl.Int64,
    "exit_reason": pl.Utf8,
    "win": pl.Boolean,
    "feat_vwap_extension": pl.Float64,
    "feat_volume_ratio": pl.Float64,
    "feat_atr_pct": pl.Float64,
    "feat_time_of_day_min": pl.Float64,
    "feat_day_of_week": pl.Float64,
    "feat_factors_count": pl.Float64,
}


def _journal_dir() -> Path:
    return Path(os.environ.get("TRADE_JOURNAL_DIR", "data/trade_journal"))


def _partition_path(d: date) -> Path:
    return _journal_dir() / f"{d.isoformat()}.parquet"


def append_trade(record: dict) -> Path:
    """Append a single trade to its date-partitioned parquet file."""
    missing = set(TRADE_SCHEMA.keys()) - set(record.keys())
    if missing:
        raise ValueError(f"trade record missing required fields: {sorted(missing)}")

    d = record["entry_time"].date() if isinstance(record["entry_time"], datetime) else record["entry_time"]
    path = _partition_path(d)
    path.parent.mkdir(parents=True, exist_ok=True)

    new_row = pl.DataFrame([record], schema=TRADE_SCHEMA)

    with _LOCK:
        if path.exists():
            existing = pl.read_parquet(path)
            out = pl.concat([existing, new_row], how="vertical_relaxed")
        else:
            out = new_row
        out.write_parquet(path)
    return path


def read_trades(
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
) -> pl.DataFrame:
    """Read all trades in [date_from, date_to] (inclusive). Either bound may be None."""
    d = _journal_dir()
    if not d.exists():
        return pl.DataFrame(schema=TRADE_SCHEMA)

    files = sorted(d.glob("*.parquet"))
    if not files:
        return pl.DataFrame(schema=TRADE_SCHEMA)

    def _in_range(p: Path) -> bool:
        try:
            file_date = date.fromisoformat(p.stem)
        except ValueError:
            return False
        if date_from and file_date < date_from:
            return False
        if date_to and file_date > date_to:
            return False
        return True

    selected = [p for p in files if _in_range(p)]
    if not selected:
        return pl.DataFrame(schema=TRADE_SCHEMA)
    return pl.concat([pl.read_parquet(p) for p in selected], how="vertical_relaxed")
