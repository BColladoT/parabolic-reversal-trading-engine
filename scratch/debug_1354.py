#!/usr/bin/env python3
"""Check 13:54 bar data."""
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

print("Bar data from bar_df:")
for row in bar_df.to_dicts():
    ts = row['timestamp'].astimezone(et_tz)
    if ts.hour == 13 and 52 <= ts.minute <= 56:
        print(f"{ts.strftime('%H:%M')}: Open={row['open']:.2f}, High={row['high']:.2f}, Low={row['low']:.2f}, Close={row['close']:.2f}")

print()
print("Ticks around 13:54:")
for row in tick_df.filter(tick_df['trade_price'] > 0).to_dicts():
    ts = row['timestamp'].astimezone(et_tz)
    if ts.hour == 13 and 52 <= ts.minute <= 56:
        print(f"{ts.strftime('%H:%M:%S')}: ${row['trade_price']:.2f}")
