#!/usr/bin/env python3
"""
Scan extended micro-cap universe (1100+ stocks) for parabolic moves.
"""
import os
from datetime import datetime
from pathlib import Path

from src.backtest.historical_screener import historical_screener
from src.backtest.extended_universe import ALL_MICRO_CAP_SYMBOLS

print("="*80)
print("EXTENDED MICRO-CAP UNIVERSE SCAN")
print(f"Scanning {len(ALL_MICRO_CAP_SYMBOLS)} symbols")
print("="*80)

# Clear old cache to force fresh scan
cache_file = Path("data/cache/setups/setups_20200101_20211231.pkl")
if cache_file.exists():
    print(f"\nRemoving old cache: {cache_file}")
    os.remove(cache_file)

print("\nScanning 2020-2021 (Meme Stock Era)...")
print("This will take several minutes with 1100+ symbols...\n")

setups = historical_screener.scan_for_parabolic_setups(
    symbols=ALL_MICRO_CAP_SYMBOLS,
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2021, 12, 31),
    min_gain_percent=30.0,  # Lower threshold for more results
    use_cache=True
)

print(f"\n{'='*80}")
print(f"TOTAL PARABOLIC SETUPS FOUND: {len(setups)}")
print(f"{'='*80}\n")

if setups:
    print(f"{'Date':<12} {'Symbol':<8} {'Gain':<10} {'Close':<10} {'Volume':<15} {'Days'}")
    print("-" * 80)
    for s in setups[:50]:
        date_str = s.date.strftime('%Y-%m-%d')
        print(f"{date_str:<12} {s.symbol:<8} {s.gain_percent:>8.1f}%  ${s.day_close:<9.2f} "
              f"{s.day_volume:>14,}  {s.days_up}")
    
    if len(setups) > 50:
        print(f"\n... and {len(setups) - 50} more setups")
    
    # Export
    csv_path = historical_screener.export_setups_for_backtest(setups)
    print(f"\nExported to: {csv_path}")
    
    # Show some test commands
    print(f"\nTest specific setups:")
    for s in setups[:5]:
        print(f"  python run_historical_backtest.py --symbol {s.symbol} --date {s.date.strftime('%Y-%m-%d')}")
