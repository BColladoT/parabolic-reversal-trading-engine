#!/usr/bin/env python3
"""Debug validation."""
import sys
sys.path.insert(0, 'src')

import polars as pl
from pathlib import Path
from datetime import datetime
import pytz

symbol = 'IGC'
date_str = '2020-08-12'
parquet_dir = Path('data/cache/1min_extended')

print(f"Testing validation for {symbol} on {date_str}")

# Find data file (same logic as provider)
data_file = parquet_dir / f"{symbol}.parquet"
if not data_file.exists():
    matching_files = list(parquet_dir.glob(f"{symbol}_1min_*.parquet"))
    if matching_files:
        data_file = matching_files[0]
        print(f"Found: {data_file}")
    else:
        print("No file found!")
        sys.exit(1)
else:
    print(f"Found: {data_file}")

# Load
df = pl.read_parquet(data_file)
print(f"Total rows: {len(df)}")

# Filter to date
date_val = datetime.strptime(date_str, '%Y-%m-%d').date()
df = df.filter(pl.col('timestamp').dt.date() == date_val)
print(f"Rows for date: {len(df)}")

if len(df) == 0:
    print("No data for date!")
    sys.exit(1)

# Calculate VWAP from market open
et_tz = pytz.timezone('America/New_York')
timestamps = df['timestamp'].to_list()

market_open_idx = 0
for i, ts in enumerate(timestamps):
    ts_et = ts.astimezone(et_tz) if ts.tzinfo else et_tz.localize(ts)
    if ts_et.hour >= 9 and ts_et.minute >= 30:
        market_open_idx = i
        break

print(f"Market open at index: {market_open_idx}")

# Calculate VWAP
df = df.with_columns([
    ((pl.col('high') + pl.col('low') + pl.col('close')) / 3).alias('typical_price')
])

df_list = df.to_dicts()
cum_pv = 0.0
cum_vol = 0.0
vwap_values = []

for i, row in enumerate(df_list):
    if i >= market_open_idx:
        cum_pv += row['typical_price'] * row['volume']
        cum_vol += row['volume']
        if cum_vol > 0:
            vwap_values.append(cum_pv / cum_vol)
        else:
            vwap_values.append(row['close'])
    else:
        vwap_values.append(row['close'])

df = df.with_columns([pl.Series('vwap', vwap_values)])
df = df.drop(['typical_price'])

# Calculate deviation
df = df.with_columns([
    ((pl.col('close') - pl.col('vwap')) / pl.col('vwap') * 100).alias('vwap_dev')
])

max_dev = df['vwap_dev'].abs().max()
print(f"Max VWAP deviation: {max_dev:.2f}%")
print(f"Passes 23% threshold: {max_dev >= 23.0}")
