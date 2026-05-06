#!/usr/bin/env python3
import polars as pl
from pathlib import Path
from datetime import datetime

# Load a sample parquet file
parquet_path = Path("data/cache/VCIG_1min_20190101_20241231.parquet")
if not parquet_path.exists():
    print(f"File not found: {parquet_path}")
    # List available files
    cache_dir = Path("data/cache")
    files = list(cache_dir.glob("*_1min_*.parquet"))[:5]
    print(f"Available files: {[f.name for f in files]}")
else:
    df = pl.read_parquet(parquet_path)
    print(f"Columns: {df.columns}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nLast few rows:")
    print(df.tail())
    
    # Check date filtering
    timestamp_col = df.columns[0]  # Usually first column
    print(f"\nTimestamp column: {timestamp_col}")
    print(f"Type: {df[timestamp_col].dtype}")
    
    # Try to filter for a specific date
    date = datetime(2024, 11, 27)
    df_with_date = df.with_columns([
        pl.col(timestamp_col).dt.date().alias('_date')
    ])
    filtered = df_with_date.filter(pl.col('_date') == date.date())
    print(f"\nFiltered rows for {date.date()}: {len(filtered)}")
    if len(filtered) > 0:
        print(filtered.head())
