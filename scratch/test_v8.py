#!/usr/bin/env python3
"""Test V8 EARLY ENTRY strategy - restricts entry to 9:45-11:30 AM only."""
import sys
sys.path.insert(0, '.')

from datetime import datetime
import pytz

from src.backtest.tick_backtest_engine_v8 import TickBacktestEngineV8

test_cases = [
    ("WWR", "2020-10-05", "Big winner (parabolic 150%)"),
    ("AMC", "2021-06-02", "Meme squeeze (+98%)"),
    ("RENT", "2024-04-11", "Big loser (parabolic 150%)"),
    ("GME", "2021-01-27", "Gamma squeeze (+8% only)"),
    ("SIDU", "2022-06-15", "Recent winner (+70%)"),
]

engine = TickBacktestEngineV8()
et_tz = pytz.timezone('America/New_York')

print("\n" + "="*70)
print("TESTING V8 ENGINE (EARLY ENTRY)")
print("Entry: 9:45-11:30 AM only (avoids late bag holding)")
print("="*70)

total_pnl = 0
total_trades = 0
for symbol, date_str, note in test_cases:
    date = datetime.strptime(date_str, "%Y-%m-%d")
    date = et_tz.localize(date)
    
    print("\n" + "="*70)
    print(f"V8 EARLY: {symbol} on {date_str}")
    print(f"Note: {note}")
    print("="*70)
    
    result = engine.run_tick_backtest(symbol, date, verbose=True)
    total_pnl += result.total_pnl
    total_trades += result.total_trades

print("\n" + "="*70)
print(f"TOTAL: {total_trades} trades, ${total_pnl:+.2f}")
print("="*70)
