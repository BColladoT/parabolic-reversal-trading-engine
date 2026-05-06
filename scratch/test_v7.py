#!/usr/bin/env python3
"""Test V7 OPTIMAL strategy - best of V5 entry + V6 risk management."""
import sys
sys.path.insert(0, '.')

from datetime import datetime
import pytz

from src.backtest.tick_backtest_engine_v7 import TickBacktestEngineV7

test_cases = [
    ("WWR", "2020-10-05", "Big winner (parabolic 150%)"),
    ("AMC", "2021-06-02", "Meme squeeze (+98%)"),
    ("RENT", "2024-04-11", "Big loser (parabolic 150%)"),
    ("GME", "2021-01-27", "Gamma squeeze (+8% only)"),
    ("SIDU", "2022-06-15", "Recent winner (+70%)"),
]

engine = TickBacktestEngineV7()
et_tz = pytz.timezone('America/New_York')

print("\n" + "="*70)
print("TESTING V7 ENGINE (OPTIMAL)")
print("Entry: 2 of 3 criteria + V6 Risk Management (3% stop, breakeven, trailing)")
print("="*70)

total_pnl = 0
for symbol, date_str, note in test_cases:
    date = datetime.strptime(date_str, "%Y-%m-%d")
    date = et_tz.localize(date)
    
    print("\n" + "="*70)
    print(f"V7 OPTIMAL: {symbol} on {date_str}")
    print(f"Strategy: 2/3 criteria (VWAP>1.15x, Vol<70%, Prox>93%) + V6 Risk")
    print(f"Note: {note}")
    print("="*70)
    
    result = engine.run_tick_backtest(symbol, date, verbose=True)
    total_pnl += result.total_pnl

print("\n" + "="*70)
print(f"TOTAL: 5 tests, ${total_pnl:+.2f}")
print("="*70)
