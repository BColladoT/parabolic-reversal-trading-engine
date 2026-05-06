#!/usr/bin/env python3
import polars as pl
from pathlib import Path
from datetime import datetime

symbol = "FFAI"
date_str = "2024-05-16"
file_path = Path(f"data/cache/{symbol}_1min_20190101_20241231.parquet")

print(f"Reading: {file_path}")
print(f"File exists: {file_path.exists()}")

df = pl.read_parquet(file_path)
print(f"\nTotal rows: {len(df)}")
print(f"Columns: {df.columns}")

# Find timestamp column
ts_col = None
for col in df.columns:
    if 'time' in col.lower() or 'date' in col.lower():
        ts_col = col
        break

print(f"\nTimestamp column: {ts_col}")
print(f"Timestamp dtype: {df[ts_col].dtype}")

# Show first few timestamps
print(f"\nFirst 5 timestamps:")
print(df[ts_col].head())

# Try to convert
print(f"\nConverting timestamps...")
if df[ts_col].dtype == pl.Utf8:
    df = df.with_columns([
        pl.col(ts_col).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False).alias('_ts')
    ])
else:
    # Handle datetime with timezone
    df = df.with_columns([
        pl.col(ts_col).dt.replace_time_zone(None).alias('_ts')
    ])

print(f"Converted. First 5 _ts:")
print(df['_ts'].head())

# Extract date
df = df.with_columns([pl.col('_ts').dt.date().alias('_date')])
print(f"\nDate range in file:")
print(f"  Min: {df['_date'].min()}")
print(f"  Max: {df['_date'].max()}")
print(f"  Unique dates: {df['_date'].n_unique()}")

# Filter
target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
print(f"\nFiltering for: {target_date}")
df_filtered = df.filter(pl.col('_date') == target_date)
print(f"Filtered rows: {len(df_filtered)}")

if len(df_filtered) > 0:
    print(f"\nSample data:")
    print(df_filtered[['_ts', 'open', 'high', 'low', 'close', 'volume']].head(10))
else:
    print("\nNo rows found! Checking available dates near target...")
    all_dates = df['_date'].unique().to_list()
    all_dates.sort()
    target_idx = None
    for i, d in enumerate(all_dates):
        if d == target_date:
            target_idx = i
            break
    if target_idx is not None:
        start = max(0, target_idx - 3)
        end = min(len(all_dates), target_idx + 4)
        print(f"Dates around target: {all_dates[start:end]}")
