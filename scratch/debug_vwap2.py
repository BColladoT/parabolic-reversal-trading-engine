#!/usr/bin/env python3
"""Debug VWAP calculation in V5."""
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

# Replicate V5 bar building EXACTLY
bars = []
cumulative_tp_v = 0.0
cumulative_vol = 0.0
day_high = 0.0
day_open = 0.0

# Convert to list and sort by timestamp to ensure consistent ordering
bar_list = sorted(bar_df.to_dicts(), key=lambda x: x['timestamp'])
print(f"Processing {len(bar_list)} bars...")

for i, row in enumerate(bar_list):
    ts = row['timestamp']
    ts_et = ts.astimezone(et_tz) if ts.tzinfo else ts
    
    typical_price = (row['high'] + row['low'] + row['close']) / 3
    cumulative_tp_v += typical_price * row['volume']
    cumulative_vol += row['volume']
    vwap = cumulative_tp_v / cumulative_vol if cumulative_vol > 0 else row['close']
    
    if day_open == 0:
        day_open = row['open']
    if row['high'] > day_high:
        day_high = row['high']
    
    bars.append({
        'timestamp': ts,
        'time_et': ts_et,
        'open': row['open'],
        'high': row['high'],
        'low': row['low'],
        'close': row['close'],
        'volume': row['volume'],
        'vwap': vwap,
        'day_open': day_open,
        'day_high': day_high
    })
    
    if ts_et.hour == 13 and 52 <= ts_et.minute <= 56:
        print(f"bars[{i}]: {ts_et.strftime('%H:%M')}: close={row['close']:.2f}, vwap={vwap:.2f}")

print(f"\nTotal bars built: {len(bars)}")
print("\nSearching for 13:54 in final bars list:")
for i, bar in enumerate(bars):
    if bar['time_et'].hour == 13 and bar['time_et'].minute == 54:
        print(f"Found at index {i}: {bar['time_et'].strftime('%H:%M')}: close={bar['close']:.2f}")
