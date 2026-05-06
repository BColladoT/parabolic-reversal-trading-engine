#!/usr/bin/env python3
"""Simple test of backtest functionality."""
import sys
from datetime import datetime

from src.backtest.backtest_engine import backtest_engine
from src.backtest.data_fetcher import data_fetcher
from src.backtest.visualizer import visualizer

print("="*60)
print("BACKTEST ENGINE TEST")
print("="*60)

# Test 1: Find candidates
print("\n1. Testing data fetcher...")
symbols = ['AAPL', 'TSLA']
candidates = data_fetcher.find_parabolic_candidates(symbols, lookback_days=7, min_gain_percent=5.0)
print(f"   Found {len(candidates)} candidates")

# Test 2: Run a backtest (recent date with known volatility)
print("\n2. Running sample backtest...")
test_date = datetime(2025, 3, 3)  # Recent Monday

result = backtest_engine.run_backtest(
    symbol='TSLA',
    date=test_date,
    verbose=True
)

print("\n" + "="*60)
print(f"RESULTS:")
print(f"  Total Trades: {result.total_trades}")
print(f"  Win Rate: {result.win_rate:.1%}")
print(f"  Total P&L: ${result.total_pnl:+.2f}")
print("="*60)

# Test 3: Generate HTML report if trades were made
if result.audit_records:
    print("\n3. Generating HTML report...")
    report_path = visualizer.generate_html_report(result)
    print(f"   Report saved to: {report_path}")
else:
    print("\n3. No trades to report")

print("\nBacktest test complete!")
