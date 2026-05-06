#!/usr/bin/env python3
"""
Scan the meme stock era (2020-2021) for massive parabolic moves.
This was when GME, AMC, and others had historic runs.
"""
from datetime import datetime
from src.backtest.historical_screener import historical_screener

print("="*70)
print("SCANNING MEME STOCK ERA (2020-2021)")
print("="*70)

symbols = historical_screener.load_micro_cap_universe()

# Scan 2020-2021 (meme stock peak)
setups = historical_screener.scan_for_parabolic_setups(
    symbols=symbols,
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2021, 12, 31),
    min_gain_percent=30.0,  # Lower threshold for more results
    use_cache=True
)

print(f"\n{'='*70}")
print(f"TOTAL PARABOLIC SETUPS FOUND: {len(setups)}")
print(f"{'='*70}\n")

if setups:
    # Show all
    print(f"{'Date':<12} {'Symbol':<8} {'Gain':<10} {'Close':<10} {'Volume':<15} {'Days'}")
    print("-" * 70)
    for s in setups[:50]:  # Show first 50
        date_str = s.date.strftime('%Y-%m-%d')
        print(f"{date_str:<12} {s.symbol:<8} {s.gain_percent:>8.1f}%  ${s.day_close:<9.2f} "
              f"{s.day_volume:>14,}  {s.days_up}")
    
    if len(setups) > 50:
        print(f"\n... and {len(setups) - 50} more setups")
    
    # Export
    csv_path = historical_screener.export_setups_for_backtest(setups)
    print(f"\nExported to: {csv_path}")
    
    print(f"\nTo test a specific setup:")
    print(f"  python run_historical_backtest.py --symbol AMC --date 2021-01-27")
else:
    print("No setups found in this period.")
