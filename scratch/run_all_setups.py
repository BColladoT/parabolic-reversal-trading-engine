#!/usr/bin/env python3
"""
Test ALL parabolic setups without strict First Red Day filtering.
This will give you many more setups to backtest.
"""
import pickle
from datetime import datetime
from pathlib import Path

from src.backtest.tick_backtest_engine import tick_backtest_engine
from src.backtest.historical_screener import historical_screener
from src.backtest.extended_universe import ALL_MICRO_CAP_SYMBOLS

print("="*80)
print("TESTING ALL PARABOLIC SETUPS (Relaxed Filtering)")
print("="*80)

# Clear cache to force fresh scan
import os
cache_file = Path("data/cache/setups/setups_20190101_20241231.pkl")
if cache_file.exists():
    print(f"\nClearing old cache...")
    os.remove(cache_file)

# Scan with RELAXED criteria (no strict First Red Day filter)
print(f"\nScanning {len(ALL_MICRO_CAP_SYMBOLS)} symbols (2019-2024)...")
print("Criteria: 30%+ gain (relaxed from 50%)")
print("No strict multi-day filter\n")

all_setups = historical_screener.scan_for_parabolic_setups(
    symbols=ALL_MICRO_CAP_SYMBOLS,
    start_date=datetime(2019, 1, 1),
    end_date=datetime(2024, 12, 31),
    min_gain_percent=30.0,  # Relaxed from 50%
    max_gain_percent=1000.0,  # Allow extreme moves
    min_volume_multiplier=2.0,  # Relaxed from 3x
    use_cache=True
)

print(f"\n{'='*80}")
print(f"TOTAL SETUPS FOUND: {len(all_setups)}")
print(f"{'='*80}\n")

if not all_setups:
    print("No setups found!")
    exit()

# Show all setups
print(f"{'#':<4} {'Date':<12} {'Symbol':<8} {'Gain':<10} {'Price':<10} {'Volume':<15} {'Days'}")
print("-" * 80)
for i, s in enumerate(all_setups[:100], 1):  # Show first 100
    date_str = s.date.strftime('%Y-%m-%d')
    print(f"{i:<4} {date_str:<12} {s.symbol:<8} {s.gain_percent:>8.1f}%  ${s.day_close:<9.2f} "
          f"{s.day_volume:>14,}  {s.days_up}")

if len(all_setups) > 100:
    print(f"\n... and {len(all_setups) - 100} more setups")

# Now test each setup
print(f"\n{'='*80}")
print("BACKTESTING ALL SETUPS")
print(f"{'='*80}\n")

print(f"{'Date':<12} {'Symbol':<8} {'Gain':<10} {'Entry':<10} {'Exit':<10} {'P&L':<12} {'Result'}")
print("-" * 80)

total_pnl = 0
total_trades = 0
setups_with_trades = 0

for i, setup in enumerate(all_setups):
    result = tick_backtest_engine.run_tick_backtest(
        symbol=setup.symbol,
        date=setup.date,
        verbose=False  # Suppress individual output
    )
    
    # Get entry/exit info
    entry_price = "-"
    exit_price = "-"
    pnl_str = "$0.00"
    
    for audit in result.audit_records:
        if audit.action.value == 'entry':
            entry_price = f"${audit.price:.2f}"
        elif audit.action.value == 'exit' and audit.pnl is not None:
            exit_price = f"${audit.exit_price:.2f}" if audit.exit_price else "-"
            pnl_str = f"${audit.pnl:+.2f}"
            total_pnl += audit.pnl
    
    if result.total_trades > 0:
        setups_with_trades += 1
        total_trades += result.total_trades
        status = "TRADE"
    else:
        status = "SKIP"
    
    date_str = setup.date.strftime('%Y-%m-%d')
    print(f"{date_str:<12} {setup.symbol:<8} {setup.gain_percent:>8.1f}%  "
          f"{entry_price:<10} {exit_price:<10} {pnl_str:<12} {status}")

# Summary
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Total Setups:        {len(all_setups)}")
print(f"Setups with Trades:  {setups_with_trades}")
print(f"Total Trades:        {total_trades}")
print(f"Total P&L:           ${total_pnl:+.2f}")
print(f"{'='*80}")

# Export results
import csv
output_file = "reports/all_setups_backtest.csv"
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Date', 'Symbol', 'Gain%', 'Price', 'Volume', 'DaysUp', 
                     'EntryPrice', 'ExitPrice', 'P&L', 'Trades'])
    
    for setup in all_setups:
        result = tick_backtest_engine.run_tick_backtest(
            symbol=setup.symbol,
            date=setup.date,
            verbose=False
        )
        
        entry = ""
        exit_p = ""
        pnl = ""
        
        for audit in result.audit_records:
            if audit.action.value == 'entry':
                entry = f"{audit.price:.2f}"
            elif audit.action.value == 'exit' and audit.pnl is not None:
                exit_p = f"{audit.exit_price:.2f}" if audit.exit_price else ""
                pnl = f"{audit.pnl:.2f}"
        
        writer.writerow([
            setup.date.strftime('%Y-%m-%d'),
            setup.symbol,
            setup.gain_percent,
            setup.day_close,
            setup.day_volume,
            setup.days_up,
            entry,
            exit_p,
            pnl,
            result.total_trades
        ])

print(f"\nResults exported to: {output_file}")
