#!/usr/bin/env python3
"""Final debug - check bar_df directly."""
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

print("Fetching fresh data...")
tick_df = tick_fetcher.fetch_combined_tick_data(symbol, date, use_quotes=False)
bar_df = tick_fetcher.aggregate_trades_to_bars(tick_df, interval_seconds=60)

print("\nDirect bar_df filter for 13:54:")
for row in bar_df.to_dicts():
    ts = row['timestamp'].astimezone(et_tz)
    if ts.hour == 13 and ts.minute == 54:
        print(f"{ts.strftime('%H:%M')}: Open={row['open']:.2f}, High={row['high']:.2f}, Close={row['close']:.2f}")
        break

print("\nFirst 5 bars from bar_df.to_dicts():")
for i, row in enumerate(bar_df.to_dicts()[:5]):
    ts = row['timestamp'].astimezone(et_tz)
    print(f"{i}: {ts.strftime('%H:%M')}: Close={row['close']:.2f}")

print("\nLast 5 bars from bar_df.to_dicts():")
rows = list(bar_df.to_dicts())
for i, row in enumerate(rows[-5:]):
    ts = row['timestamp'].astimezone(et_tz)
    print(f"{len(rows)-5+i}: {ts.strftime('%H:%M')}: Close={row['close']:.2f}")

print("\nSearching for 66.44 value:")
for i, row in enumerate(bar_df.to_dicts()):
    if abs(row['close'] - 66.44) < 0.01:
        ts = row['timestamp'].astimezone(et_tz)
        print(f"Found at index {i}: {ts.strftime('%H:%M')}: Close={row['close']:.2f}")
