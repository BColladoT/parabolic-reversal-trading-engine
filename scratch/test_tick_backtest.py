#!/usr/bin/env python3
"""
Test Tick-Based Backtesting with Historical Data
Fetches actual trade data from Alpaca for accurate simulation.
"""
import sys
from datetime import datetime, timedelta

from src.backtest.tick_backtest_engine import tick_backtest_engine
from src.backtest.historical_tick_fetcher import tick_fetcher
from src.backtest.visualizer import visualizer

print("="*70)
print("TICK-LEVEL BACKTEST ENGINE")
print("Using Historical Trade Data from Alpaca")
print("="*70)

# Test: TSLA on a recent volatile day
symbol = "TSLA"
test_date = datetime(2025, 3, 3)  # Recent trading day

print(f"\nSymbol: {symbol}")
print(f"Date: {test_date.date()}")
print(f"\nFetching historical tick data (this may take a moment)...")

# First, let's see what tick data looks like
sample_ticks = tick_fetcher.fetch_historical_trades(
    symbol=symbol,
    start=test_date.replace(hour=10, minute=0),
    end=test_date.replace(hour=10, minute=5),  # Just 5 minutes for preview
    use_cache=True
)

if not sample_ticks.is_empty():
    print(f"\nSample tick data (first 5 trades):")
    print(f"{'Time':<15} {'Price':<10} {'Size':<10} {'Exchange':<10}")
    print("-" * 50)
    for tick in sample_ticks.head(5).to_dicts():
        ts = tick['timestamp'].strftime('%H:%M:%S.%f')[:-3]
        print(f"{ts:<15} ${tick['trade_price']:<9.2f} {tick['trade_size']:<10} {tick['trade_exchange']:<10}")

# Run full tick-level backtest
print(f"\n{'='*70}")
print("Running Tick-Level Backtest...")
print(f"{'='*70}\n")

result = tick_backtest_engine.run_tick_backtest(
    symbol=symbol,
    date=test_date,
    verbose=True
)

# Generate HTML report if trades were made
if result.total_trades > 0:
    print(f"\n{'='*70}")
    print("Generating HTML Report...")
    print(f"{'='*70}")
    
    report_path = visualizer.generate_html_report(result)
    print(f"Report saved to: {report_path}")
    
    # Also create charts
    print("\nGenerating charts...")
    
    # Equity curve
    chart_path = f"reports/{symbol}_{test_date.strftime('%Y%m%d')}_equity_tick.png"
    visualizer.create_equity_curve(result, save_path=chart_path)
    
    print(f"\nFinal Results:")
    print(f"  Total Trades: {result.total_trades}")
    print(f"  Win Rate: {result.win_rate:.1%}")
    print(f"  Total P&L: ${result.total_pnl:+.2f}")
    print(f"  Profit Factor: {result.profit_factor:.2f}")
else:
    print(f"\nNo trades executed - no parabolic setup detected on this date.")

print(f"\n{'='*70}")
print("Tick Backtest Complete!")
print(f"{'='*70}")
