#!/usr/bin/env python3
"""Check what signals are present during 10-11 AM ET."""
from datetime import datetime
import pytz

from src.backtest.historical_tick_fetcher import tick_fetcher
from src.indicators.numba_kernels import calculate_vwap_numba, calculate_atr_numba
from src.utils.config import CONFIG

import polars as pl

symbol = "KOSS"
date = datetime(2021, 1, 27)

print(f"Checking {symbol} on {date.date()}")
print(f"Config: VWAP Ext > {CONFIG.signals.vwap_extension_threshold}x, Vol < {CONFIG.signals.volume_exhaustion_factor}x avg")
print()

# Fetch data
tick_df = tick_fetcher.fetch_combined_tick_data(symbol, date, use_quotes=False)
bar_df = tick_fetcher.aggregate_trades_to_bars(tick_df, interval_seconds=60)

# Calculate indicators
highs = bar_df['high'].to_numpy()
lows = bar_df['low'].to_numpy()
closes = bar_df['close'].to_numpy()
volumes = bar_df['volume'].to_numpy()

vwap_values = calculate_vwap_numba(highs, lows, closes, volumes)
bar_df = bar_df.with_columns([
    pl.Series('vwap', vwap_values),
    (pl.col('close') / pl.col('vwap')).alias('vwap_extension'),
])

# Convert to ET and filter 10-11 AM
et_tz = pytz.timezone('America/New_York')
bar_df = bar_df.with_columns([
    pl.col('timestamp').map_elements(lambda x: x.astimezone(et_tz) if x.tzinfo else x, return_dtype=pl.Datetime).alias('et_time')
])

# Filter to 10-11 AM ET
window_bars = bar_df.filter(
    (pl.col('et_time').dt.hour() >= 10) & (pl.col('et_time').dt.hour() < 11)
)

print(f"Bars in 10-11 AM ET window: {len(window_bars)}")
print()

if len(window_bars) > 0:
    print(f"{'Time':<10} {'Close':<10} {'VWAP':<10} {'Ext':<8} {'Volume':<12} {'Entry?'}")
    print("-" * 70)
    
    for row in window_bars.to_dicts():
        time_str = row['et_time'].strftime('%H:%M')
        ext = row['vwap_extension']
        vol = row['volume']
        
        # Check if entry criteria met
        vwap_ok = ext >= CONFIG.signals.vwap_extension_threshold
        vol_ok = vol > 10000  # Simplified
        
        entry = "YES" if (vwap_ok and vol_ok) else ""
        if not vwap_ok:
            entry = f"ext={ext:.2f}x"
        
        print(f"{time_str:<10} ${row['close']:<9.2f} ${row['vwap']:<9.2f} {ext:<7.2f}x {vol:>11,} {entry}")
    
    # Summary
    max_ext = window_bars['vwap_extension'].max()
    print()
    print(f"Max VWAP Extension in window: {max_ext:.2f}x")
    print(f"Required: {CONFIG.signals.vwap_extension_threshold}x")
    
    if max_ext < CONFIG.signals.vwap_extension_threshold:
        print()
        print(f">>> PRICE NEVER EXTENDED FAR ENOUGH ABOVE VWAP!")
        print(f">>> Max was {max_ext:.2f}x, need {CONFIG.signals.vwap_extension_threshold}x")
        print()
        print("Recommendations:")
        print("1. Lower vwap_extension_threshold to 1.05 or 1.10")
        print("2. Expand execution window earlier (9:30 AM)")
        print("3. Check full day - maybe move happened outside 10-11 AM")
        
        # Check full day max
        full_max_ext = bar_df['vwap_extension'].max()
        full_max_time = bar_df.filter(pl.col('vwap_extension') == full_max_ext)['et_time'].first()
        print()
        print(f"Full day max VWAP extension: {full_max_ext:.2f}x at {full_max_time.strftime('%H:%M') if full_max_time else 'N/A'}")
else:
    print("No bars in 10-11 AM window!")
