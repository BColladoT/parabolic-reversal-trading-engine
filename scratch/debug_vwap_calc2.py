#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

import polars as pl
from pathlib import Path
from datetime import datetime

symbol = 'IGC'
date_str = '2020-08-12'
parquet_dir = Path('data/cache/1min_extended')

matching_files = list(parquet_dir.glob(f"{symbol}_1min_*.parquet"))
data_file = matching_files[0]

df = pl.read_parquet(data_file)
date_val = datetime.strptime(date_str, '%Y-%m-%d').date()
df = df.filter(pl.col('timestamp').dt.date() == date_val)

print(f"Total rows for date: {len(df)}")
print(f"Raw timestamps (first 10):")
for i in range(min(10, len(df))):
    print(f"  {df['timestamp'][i]}")

# Add ET time
df = df.with_columns([
    pl.col('timestamp').dt.convert_time_zone('America/New_York').dt.hour().alias('et_hour'),
    pl.col('timestamp').dt.convert_time_zone('America/New_York').dt.minute().alias('et_minute')
])

print(f"\nET times (first 10):")
for i in range(min(10, len(df))):
    print(f"  {df['et_hour'][i]:02d}:{df['et_minute'][i]:02d}")

# Check the condition
minutes_from_midnight = df['et_hour'] * 60 + df['et_minute']
market_open_minutes = 9 * 60 + 30
print(f"\nMinutes from midnight (first 10): {minutes_from_midnight.head(10).to_list()}")
print(f"Market open minutes: {market_open_minutes}")
print(f"Condition result (first 10): {(minutes_from_midnight >= market_open_minutes).head(10).to_list()}")
