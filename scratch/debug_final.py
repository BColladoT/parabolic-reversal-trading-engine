#!/usr/bin/env python3
"""Final debug - check what data V5 is actually using."""
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
print(f"Got {len(tick_df)} ticks")

print("\nAggregating to bars...")
bar_df = tick_fetcher.aggregate_trades_to_bars(tick_df, interval_seconds=60)
print(f"Got {len(bar_df)} bars")

print("\nConverting to list...")
bar_list = list(bar_df.to_dicts())
print(f"List has {len(bar_list)} bars")

print("\nBuilding bars with VWAP...")
bars = []
cumulative_tp_v = 0.0
cumulative_vol = 0.0
day_high = 0.0
day_open = 0.0

for row in bar_list:
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
        'close': row['close'],
        'vwap': vwap,
        'day_open': day_open,
        'day_high': day_high
    })

print(f"Built {len(bars)} bars")

print("\nBars around 13:54:")
for bar in bars:
    if bar['time_et'].hour == 13 and 52 <= bar['time_et'].minute <= 56:
        print(f"{bar['time_et'].strftime('%H:%M')}: Close=${bar['close']:.2f}, VWAP=${bar['vwap']:.2f}")
