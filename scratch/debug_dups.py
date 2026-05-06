#!/usr/bin/env python3
"""Check for duplicate timestamps."""
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

tick_df = tick_fetcher.fetch_combined_tick_data(symbol, date, use_quotes=False)
bar_df = tick_fetcher.aggregate_trades_to_bars(tick_df, interval_seconds=60)

print("All timestamps with 13:54:")
for i, row in enumerate(bar_df.to_dicts()):
    ts = row['timestamp'].astimezone(et_tz)
    if ts.hour == 13 and ts.minute == 54:
        print(f"  Index {i}: {ts.strftime('%H:%M:%S.%f')}: close={row['close']:.2f}")

print("\nAll timestamps with 07:54:")
for i, row in enumerate(bar_df.to_dicts()):
    ts = row['timestamp'].astimezone(et_tz)
    if ts.hour == 7 and ts.minute == 54:
        print(f"  Index {i}: {ts.strftime('%H:%M:%S.%f')}: close={row['close']:.2f}")

print("\nSorted list - first 30:")
bar_list = sorted(bar_df.to_dicts(), key=lambda x: x['timestamp'])
for i, row in enumerate(bar_list[:30]):
    ts = row['timestamp'].astimezone(et_tz)
    print(f"  {i}: {ts.strftime('%H:%M:%S.%f')}: close={row['close']:.2f}")
