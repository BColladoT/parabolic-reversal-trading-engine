#!/usr/bin/env python3
"""Debug VWAP validation."""
import sys
sys.path.insert(0, 'src')

import polars as pl
from pathlib import Path

symbol = 'IGC'
date_str = '2020-08-12'
parquet_dir = Path('data/cache/1min_extended')

# Find data file
data_file = parquet_dir / f"{symbol}.parquet"
if not data_file.exists():
    matching_files = list(parquet_dir.glob(f"{symbol}_1min_*.parquet"))
    if matching_files:
        data_file = matching_files[0]
    else:
        print(f"No data file found for {symbol}")
        sys.exit(1)

print(f"Loading: {data_file}")
df = pl.read_parquet(data_file)

# Filter to date
from datetime import datetime
date_val = datetime.strptime(date_str, '%Y-%m-%d').date()
df = df.filter(pl.col('timestamp').dt.date() == date_val)

print(f"Rows for {date_str}: {len(df)}")

if len(df) == 0:
    print("No data for this date!")
    sys.exit(1)

# Always recalculate VWAP from market open
print("Recalculating VWAP from market open...")
import pytz

et_tz = pytz.timezone('America/New_York')
timestamps = df['timestamp'].to_list()

# Find first bar at or after 9:30 AM ET
market_open_idx = 0
for i, ts in enumerate(timestamps):
    ts_et = ts.astimezone(et_tz) if ts.tzinfo else et_tz.localize(ts)
    if ts_et.hour >= 9 and ts_et.minute >= 30:
        market_open_idx = i
        break

print(f"Market open at bar index: {market_open_idx}")

# Calculate VWAP from market open
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

df = df.with_columns([
    pl.Series('vwap', vwap_values)
])

df = df.drop(['typical_price'])

# Calculate VWAP deviation
df = df.with_columns([
    ((pl.col('close') - pl.col('vwap')) / pl.col('vwap') * 100).alias('vwap_dev')
])

print(f"VWAP deviation stats:")
print(f"  Min: {df['vwap_dev'].min():.2f}%")
print(f"  Max: {df['vwap_dev'].max():.2f}%")
print(f"  Mean: {df['vwap_dev'].mean():.2f}%")

max_dev = df['vwap_dev'].abs().max()
print(f"  Max absolute: {max_dev:.2f}%")
print(f"  Passes 23% threshold: {max_dev >= 23.0}")

# Show first few bars with VWAP
print("\nFirst 5 bars:")
print(df.head(5).select(['timestamp', 'close', 'vwap', 'vwap_dev']))

# Show bars with high VWAP deviation
high_dev = df.filter(pl.col('vwap_dev').abs() > 20)
print(f"\nBars with |VWAP dev| > 20%: {len(high_dev)}")
if len(high_dev) > 0:
    print(high_dev.select(['timestamp', 'close', 'vwap', 'vwap_dev']).head(10))
