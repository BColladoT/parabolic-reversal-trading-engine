#!/usr/bin/env python3
"""Debug cache vs fresh data."""
import sys
sys.path.insert(0, '.')
from datetime import datetime
import pytz
from src.backtest.historical_tick_fetcher import tick_fetcher

et_tz = pytz.timezone('America/New_York')

symbol = "AMC"
date_str = "2021-06-02"

date = datetime.strptime(date_str, "%Y-%m-%d")
date = et_tz.localize(date)

# Fetch with cache (default)
print("Fetching with cache (default)...")
tick_df = tick_fetcher.fetch_combined_tick_data(symbol, date, use_quotes=False)
bar_df = tick_fetcher.aggregate_trades_to_bars(tick_df, interval_seconds=60)

# Check bar at 13:54
for row in bar_df.to_dicts():
    ts = row['timestamp'].astimezone(et_tz)
    if ts.hour == 13 and ts.minute == 54:
        print(f"  13:54: Open={row['open']:.2f}, High={row['high']:.2f}, Close={row['close']:.2f}")
        break

# Now replicate what V5 does with sorting
print("\nAfter sorting (like V5 does):")
bar_list = sorted(bar_df.to_dicts(), key=lambda x: x['timestamp'])
for row in bar_list:
    ts = row['timestamp'].astimezone(et_tz)
    if ts.hour == 13 and ts.minute == 54:
        print(f"  13:54: Open={row['open']:.2f}, High={row['high']:.2f}, Close={row['close']:.2f}")
        break
