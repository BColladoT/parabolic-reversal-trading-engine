#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

import polars as pl
from pathlib import Path
from datetime import datetime

symbol = 'IGC'
date_str = '2020-08-12'
parquet_dir = Path('data/cache/1min_extended')

# Find file
matching_files = list(parquet_dir.glob(f"{symbol}_1min_*.parquet"))
data_file = matching_files[0]

print(f"Loading: {data_file}")
df = pl.read_parquet(data_file)

# Filter to date and market hours
date_val = datetime.strptime(date_str, '%Y-%m-%d').date()
df = df.filter(
    (pl.col('timestamp').dt.date() == date_val) &
    (pl.col('timestamp').dt.hour() >= 9) &
    (pl.col('timestamp').dt.hour() <= 16)
)

print(f"Rows: {len(df)}")
print(f"Timestamp sample: {df['timestamp'][0]}")

# Add ET time
df = df.with_columns([
    pl.col('timestamp').dt.convert_time_zone('America/New_York').dt.hour().alias('et_hour'),
    pl.col('timestamp').dt.convert_time_zone('America/New_York').dt.minute().alias('et_minute')
])

print(f"ET hour range: {df['et_hour'].min()} - {df['et_hour'].max()}")
print(f"ET minute range: {df['et_minute'].min()} - {df['et_minute'].max()}")

# Check after_open mask
df = df.with_columns([
    ((pl.col('et_hour') * 60 + pl.col('et_minute')) >= (9 * 60 + 30)).alias('after_open')
])

print(f"After open: {df['after_open'].sum()} / {len(df)}")
print(f"First 5 after_open values: {df['after_open'].head(5).to_list()}")
