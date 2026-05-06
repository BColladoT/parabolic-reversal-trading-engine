#!/usr/bin/env python3
"""
Diagnose why no trades are executing in backtest.
"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.backtest.historical_tick_fetcher import tick_fetcher
from src.indicators.numba_kernels import calculate_vwap_numba, calculate_atr_numba
from src.utils.config import CONFIG

import polars as pl

print("="*80)
print("BACKTEST DIAGNOSTIC - Why No Trades?")
print("="*80)

# Test a known parabolic day
symbol = "KOSS"
date = datetime(2021, 1, 27)

print(f"\nTesting: {symbol} on {date.date()}")
print(f"This was a +232% parabolic move\n")

# Fetch data
tick_df = tick_fetcher.fetch_combined_tick_data(symbol, date, use_quotes=False)

if tick_df.is_empty():
    print("ERROR: No tick data!")
    exit()

print(f"Loaded {len(tick_df)} trades")

# Show first and last timestamps
first_tick = tick_df['timestamp'].first()
last_tick = tick_df['timestamp'].last()
print(f"First trade: {first_tick}")
print(f"Last trade: {last_tick}")

# Aggregate to bars
bar_df = tick_fetcher.aggregate_trades_to_bars(tick_df, interval_seconds=60)
print(f"\nAggregated to {len(bar_df)} 1-minute bars")

# Calculate indicators
highs = bar_df['high'].to_numpy()
lows = bar_df['low'].to_numpy()
closes = bar_df['close'].to_numpy()
volumes = bar_df['volume'].to_numpy()

vwap_values = calculate_vwap_numba(highs, lows, closes, volumes)
atr_values = calculate_atr_numba(highs, lows, closes, period=14)

bar_df = bar_df.with_columns([
    pl.Series('vwap', vwap_values),
    pl.Series('atr', atr_values),
    (pl.col('close') / pl.col('vwap')).alias('vwap_extension'),
])

# Show all hours available
print(f"\n{'='*80}")
print("TRADING HOURS IN DATA")
print(f"{'='*80}")

hours = bar_df.with_columns([
    pl.col('timestamp').dt.hour().alias('hour')
]).group_by('hour').agg([
    pl.count().alias('bars'),
    pl.first('timestamp').alias('first_time')
]).sort('hour')

print(f"{'Hour':<8} {'Bars':<8} {'Sample Time'}")
print("-" * 40)
for row in hours.to_dicts():
    print(f"{row['hour']:<8} {row['bars']:<8} {row['first_time']}")

# Check execution window bars
print(f"\n{'='*80}")
print("EXECUTION WINDOW (10:00-11:00 AM)")
print(f"{'='*80}")

# Convert timestamp to hour for filtering
bar_df = bar_df.with_columns([
    pl.col('timestamp').dt.hour().alias('hour'),
    pl.col('timestamp').dt.minute().alias('minute')
])

execution_bars = bar_df.filter(
    ((pl.col('hour') == 10)) | 
    ((pl.col('hour') == 11) & (pl.col('minute') == 0))
)

print(f"Bars in 10:00-11:00 window: {len(execution_bars)}")

if len(execution_bars) > 0:
    print(f"\n{'Time':<10} {'Close':<10} {'VWAP':<10} {'Ext':<8} {'Volume':<12}")
    print("-" * 60)
    for row in execution_bars.to_dicts():
        time_str = row['timestamp'].strftime('%H:%M')
        print(f"{time_str:<10} ${row['close']:<9.2f} ${row['vwap']:<9.2f} {row['vwap_extension']:<7.2f}x {row['volume']:>11,}")
else:
    print("\n!!! NO BARS IN 10-11 AM WINDOW !!!")
    print("\nThis means the stock didn't have trading activity during this time.")
    print("Possible reasons:")
    print("1. Different timezone (data might be in UTC)")
    print("2. Trading halt/suspension")
    print("3. Different market hours")
    
    # Show bars around market open
    print(f"\nBars around 9:30 AM open:")
    morning = bar_df.filter(pl.col('hour').is_in([9, 10, 11, 12, 13, 14, 15]))
    if len(morning) > 0:
        print(f"First bar: {morning['timestamp'].first()} @ ${morning['close'].first():.2f}")
        print(f"Max extension: {morning['vwap_extension'].max():.2f}x")

print(f"\n{'='*80}")
print("RECOMMENDATION")
print(f"{'='*80}")
print("The execution window (10-11 AM ET) might be wrong for this data.")
print("Check if timestamps are in UTC (would be 3-4 hours ahead).")
print("\nTo fix, either:")
print("1. Adjust timezone handling in code")
print("2. Expand execution window to catch the move")
print("3. Check what time the parabolic move actually happened")
