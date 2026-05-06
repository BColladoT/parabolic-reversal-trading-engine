"""Build a small sample dataset for the quickstart example backtest.

Slices each chosen symbol's full 1-min parquet down to a single 30-day window
that contains a known V5 Relaxed setup. Total payload < 20 MB so it can be
committed to the repo.

Inputs:  data/cache/1min_extended/{SYMBOL}_1min_20190101_20241231.parquet
Outputs: data/sample/{SYMBOL}.parquet
"""
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "cache" / "1min_extended"
OUT = ROOT / "data" / "sample"
OUT.mkdir(parents=True, exist_ok=True)

# Symbol -> (window_start, window_end). Each window covers a known V5 Relaxed setup
# observed in reports/cached_parallel_backtest/combined_trades.csv.
WINDOWS = {
    "GME":  ("2021-01-15", "2021-02-13"),  # Jan 25 + Jan 28 wins, Jan 26 + Feb 5 losses
    "BBIG": ("2021-08-17", "2021-09-15"),  # Sep 1 win
    "KOSS": ("2021-01-15", "2021-02-13"),  # Jan 27 loss
    "MULN": ("2022-03-01", "2022-03-30"),  # Mar 7 win
    "GFAI": ("2022-01-01", "2022-01-31"),  # Jan 14 win
}

total_rows = 0
total_bytes = 0
for symbol, (start, end) in WINDOWS.items():
    src = next(SRC.glob(f"{symbol}_1min_*.parquet"))
    df = (
        pl.read_parquet(src)
        .with_columns(pl.col("timestamp").cast(pl.Datetime))
        .filter(
            (pl.col("timestamp") >= pl.lit(start).str.to_datetime())
            & (pl.col("timestamp") < pl.lit(end).str.to_datetime())
        )
        .sort("timestamp")
    )
    out_path = OUT / f"{symbol}.parquet"
    df.write_parquet(out_path, compression="zstd")
    rows = len(df)
    size = out_path.stat().st_size
    total_rows += rows
    total_bytes += size
    print(f"{symbol:>5}  {start} -> {end}  rows={rows:>6}  size={size/1024:>6.1f} KB")

print(f"\n{'TOTAL':>5}  {' ' * 23}  rows={total_rows:>6}  size={total_bytes/1024:>6.1f} KB")
