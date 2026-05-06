#!/usr/bin/env python3
"""Check if sorting works."""
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

print("Without sort:")
for i, row in enumerate(bar_df.to_dicts()):
    if 20 <= i <= 25:
        ts = row['timestamp'].astimezone(et_tz)
        print(f"  {i}: {ts.strftime('%H:%M')}: close={row['close']:.2f}")

print("\nWith sort:")
bar_list = sorted(bar_df.to_dicts(), key=lambda x: x['timestamp'])
for i, row in enumerate(bar_list):
    if 20 <= i <= 25:
        ts = row['timestamp'].astimezone(et_tz)
        print(f"  {i}: {ts.strftime('%H:%M')}: close={row['close']:.2f}")
