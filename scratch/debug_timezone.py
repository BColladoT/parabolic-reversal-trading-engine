#!/usr/bin/env python3
"""Debug timezone handling."""
from datetime import datetime
import pytz

from src.backtest.historical_tick_fetcher import tick_fetcher
import polars as pl

symbol = "KOSS"
date = datetime(2021, 1, 27)

tick_df = tick_fetcher.fetch_combined_tick_data(symbol, date, use_quotes=False)

print("Sample timestamps from data:")
for i, row in enumerate(tick_df.head(5).to_dicts()):
    ts = row['timestamp']
    print(f"  {i+1}. {ts} (tz={ts.tzinfo})")

print()
print("Converting to ET:")
et_tz = pytz.timezone('America/New_York')

for i, row in enumerate(tick_df.head(5).to_dicts()):
    ts = row['timestamp']
    ts_et = ts.astimezone(et_tz)
    print(f"  {i+1}. UTC: {ts} -> ET: {ts_et} (hour={ts_et.hour})")

# Check all unique hours in the data
print()
print("All hours in data (UTC):")
hours = tick_df.with_columns([
    pl.col('timestamp').dt.hour().alias('hour')
]).select('hour').unique().sort('hour')

for h in hours.to_dicts():
    print(f"  UTC Hour {h['hour']}")

print()
print("Converting to ET hours:")
for h in hours.to_dicts():
    utc_hour = h['hour']
    # Sample timestamp at this hour
    sample = tick_df.filter(pl.col('timestamp').dt.hour() == utc_hour)['timestamp'].first()
    et_time = sample.astimezone(et_tz)
    print(f"  UTC Hour {utc_hour} -> ET Hour {et_time.hour}")

print()
print("Execution window 10-11 AM ET = UTC Hours 15-16 (winter)")
