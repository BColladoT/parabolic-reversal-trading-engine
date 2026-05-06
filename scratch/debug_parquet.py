#!/usr/bin/env python3
import polars as pl
from pathlib import Path
from datetime import datetime

# Find GNS file
symbol = "GNS"
date_str = "2023-01-19"

cache_dir = Path("data/cache")
pattern = f"{symbol}_1min_*.parquet"
files = list(cache_dir.glob(pattern))

if not files:
    print(f"No files found for {symbol}")
else:
    file_path = files[0]
    print(f"Reading: {file_path.name}")
    
    df = pl.read_parquet(file_path)
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Find timestamp column
    ts_col = None
    for col in df.columns:
        if 'time' in col.lower() or 'date' in col.lower():
            ts_col = col
            break
    
    print(f"\nTimestamp column: {ts_col}")
    print(f"Timestamp dtype: {df[ts_col].dtype}")
    
    # Check date range
    if ts_col:
        # Convert and check
        if df[ts_col].dtype == pl.Utf8:
            df = df.with_columns([
                pl.col(ts_col).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False).alias('_ts')
            ])
        else:
            df = df.with_columns([pl.col(ts_col).cast(pl.Datetime).alias('_ts')])
        
        df = df.with_columns([pl.col('_ts').dt.date().alias('_date')])
        
        print(f"\nDate range in file:")
        print(f"  Min: {df['_date'].min()}")
        print(f"  Max: {df['_date'].max()}")
        
        # Filter to target date
        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        filtered = df.filter(pl.col('_date') == target_date)
        print(f"\nRows for {date_str}: {len(filtered)}")
        
        if len(filtered) > 0:
            print(f"\nSample rows for {date_str}:")
            print(filtered[['_ts', 'open', 'high', 'low', 'close', 'volume']].head(10))
