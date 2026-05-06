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
df = pl.read_parquet(matching_files[0])
date_val = datetime.strptime(date_str, '%Y-%m-%d').date()
df = df.filter(pl.col('timestamp').dt.date() == date_val)

# Add ET time with explicit casting
df = df.with_columns([
    pl.col('timestamp').dt.convert_time_zone('America/New_York').dt.hour().cast(pl.Int32).alias('et_hour'),
    pl.col('timestamp').dt.convert_time_zone('America/New_York').dt.minute().cast(pl.Int32).alias('et_minute')
])

print(f"Types: hour={df['et_hour'].dtype}, minute={df['et_minute'].dtype}")
print(f"First 5 hours: {df['et_hour'].head(5).to_list()}")
print(f"First 5 minutes: {df['et_minute'].head(5).to_list()}")

# Calculate minutes from midnight manually
hours = df['et_hour'].to_numpy()
minutes = df['et_minute'].to_numpy()
minutes_from_midnight = hours * 60 + minutes

print(f"Minutes from midnight (first 5): {minutes_from_midnight[:5]}")
print(f"Market open (9*60+30): {9*60+30}")
print(f"After 9:30? (first 5): {(minutes_from_midnight >= 9*60+30)[:5]}")
