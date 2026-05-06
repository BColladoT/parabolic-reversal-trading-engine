#!/usr/bin/env python3
"""Debug the actual V5 engine bar processing."""
import sys
sys.path.insert(0, '.')
from datetime import datetime, time as dt_time
import pytz
from src.backtest.historical_tick_fetcher import tick_fetcher

et_tz = pytz.timezone('America/New_York')

symbol = "AMC"
date_str = "2021-06-02"

date = datetime.strptime(date_str, "%Y-%m-%d")
date = et_tz.localize(date)

tick_df = tick_fetcher.fetch_combined_tick_data(symbol, date, use_quotes=False)
bar_df = tick_fetcher.aggregate_trades_to_bars(tick_df, interval_seconds=60)

# Replicate V5 exactly
bars = []
cumulative_tp_v = 0.0
cumulative_vol = 0.0
day_high = 0.0
day_open = 0.0

bar_list = list(bar_df.to_dicts())
print(f"bar_list has {len(bar_list)} items")

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
    
    # Print bars around 13:54
    if ts_et.hour == 13 and 50 <= ts_et.minute <= 58:
        print(f"bar_list[{i}]: {ts_et.strftime('%H:%M')}: close={row['close']:.2f}")

print(f"\nBuilt {len(bars)} bars")
print("\nBars around 13:54 in final bars list:")
for i, bar in enumerate(bars):
    if bar['time_et'].hour == 13 and 50 <= bar['time_et'].minute <= 58:
        print(f"bars[{i}]: {bar['time_et'].strftime('%H:%M')}: close={bar['close']:.2f}, vwap={bar['vwap']:.2f}")

# Now replicate the entry logic
print("\n\n=== ENTRY LOGIC ===")
volume_history = []
best_setup = None

for i, bar in enumerate(bars):
    t = bar['time_et']
    if not isinstance(t, datetime):
        continue
    
    volume_history.append(bar['volume'])
    if len(volume_history) > 10:
        volume_history.pop(0)
    vol_peak = max(volume_history) if volume_history else bar['volume']
    
    bar['day_high'] = day_high
    
    if not (dt_time(9, 45) <= t.time() <= dt_time(14, 0)):
        continue
    
    day_gain = (bar['day_high'] - bar['day_open']) / bar['day_open'] if bar['day_open'] > 0 else 0
    if day_gain < 0.50:
        continue
    
    if bar['close'] < bar['vwap']:
        continue
    
    vwap_ext = bar['close'] / bar['vwap'] if bar['vwap'] > 0 else 1.0
    vol_ratio = bar['volume'] / vol_peak if vol_peak > 0 else 1.0
    prox = bar['close'] / bar['day_high'] if bar['day_high'] > 0 else 0
    
    criteria_met = sum([
        vwap_ext >= 1.15,
        vol_ratio <= 0.70,
        prox >= 0.93
    ])
    
    if criteria_met >= 2:
        print(f"{t.strftime('%H:%M')}: close=${bar['close']:.2f}, vwap_ext={vwap_ext:.2f}, prox={prox:.2f}, criteria={criteria_met}")
        if best_setup is None or vwap_ext > best_setup['vwap_ext']:
            best_setup = {
                'bar': bar,
                'vwap_ext': vwap_ext,
                'vol_ratio': vol_ratio,
                'prox': prox,
                'criteria': criteria_met
            }
            print(f"  -> NEW BEST at index {i}")

print()
if best_setup:
    bar = best_setup['bar']
    print(f"FINAL BEST: {bar['time_et'].strftime('%H:%M')} @ ${bar['close']:.2f}")
else:
    print("No setup found")
