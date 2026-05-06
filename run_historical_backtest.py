#!/usr/bin/env python3
"""
Historical Parabolic Reversal Backtest
Tests strategy across 6+ years of micro-cap parabolic moves.

Usage:
    python run_historical_backtest.py --years 2019-2024
    python run_historical_backtest.py --quick-test (10 setups)
    python run_historical_backtest.py --symbol TSLA --year 2023
"""
import sys
from datetime import datetime

from src.backtest.batch_backtest import batch_runner
from src.backtest.historical_screener import historical_screener
from src.backtest.tick_backtest_engine import tick_backtest_engine


def print_banner():
    print("\n" + "="*80)
    print("  PARABOLIC REVERSAL STRATEGY - HISTORICAL BACKTEST")
    print("  Testing First Red Day Strategy on Micro-Cap Stocks")
    print("  Using Alpaca Historical Tick Data (2019-2024)")
    print("="*80 + "\n")


def run_full_backtest():
    """Run comprehensive multi-year backtest."""
    print_banner()
    
    print("Configuration:")
    print("  Universe: Micro-cap stocks ($0.50-$50)")
    print("  Setup Criteria: 50%+ single day gain, 2-5 days up")
    print("  Execution Window: 10:00-11:00 AM ET")
    print("  Risk: 1% per trade, VWAP mean reversion exit")
    print()
    
    # Run the full batch backtest
    result = batch_runner.run_historical_backtest(
        start_year=2019,
        end_year=2024,
        symbols=None,  # Use default micro-cap universe
        min_gain_percent=50.0,
        max_setups=None,  # Test all found setups
        verbose=False
    )
    
    # Print final summary
    result.print_summary()
    
    print("\n" + "="*80)
    print("Reports saved to reports/ directory:")
    print("  - batch_backtest_[timestamp].csv (trade log)")
    print("  - batch_backtest_[timestamp].html (visual report)")
    print("="*80 + "\n")


def run_quick_test():
    """Quick test with limited setups."""
    print_banner()
    print("QUICK TEST MODE (10 setups)\n")
    
    result = batch_runner.run_historical_backtest(
        start_year=2023,
        end_year=2024,
        max_setups=10,
        verbose=False
    )
    
    result.print_summary()


def scan_and_list_setups():
    """Scan historical data and list all setups found."""
    print_banner()
    print("SCAN MODE: Finding all parabolic setups\n")
    
    symbols = historical_screener.load_micro_cap_universe()
    
    setups = historical_screener.scan_for_parabolic_setups(
        symbols=symbols,
        start_date=datetime(2019, 1, 1),
        end_date=datetime(2024, 12, 31),
        min_gain_percent=50.0,
        use_cache=True
    )
    
    print(f"\n{'='*80}")
    print(f"TOTAL PARABOLIC SETUPS FOUND: {len(setups)}")
    print(f"{'='*80}\n")
    
    # Show first 20
    print("First 20 setups:")
    print(f"{'Date':<12} {'Symbol':<8} {'Gain':<8} {'Price':<10} {'Volume':<15} {'Days Up'}")
    print("-" * 80)
    for s in setups[:20]:
        print(f"{s.date.strftime('%Y-%m-%d'):<12} {s.symbol:<8} "
              f"{s.gain_percent:>6.1f}%  ${s.day_close:<9.2f} "
              f"{s.day_volume:>14,}  {s.days_up}")
    
    # Analyze distribution
    analysis = historical_screener.analyze_setup_distribution(setups)
    
    print(f"\n{'='*80}")
    print("SETUP ANALYSIS:")
    print(f"{'='*80}")
    print(f"  Total Setups:           {analysis['total_setups']}")
    print(f"  Average Gain:           {analysis['avg_gain_percent']:.1f}%")
    print(f"  Median Gain:            {analysis['median_gain_percent']:.1f}%")
    print(f"  Average Volume:         {analysis['avg_volume']:,.0f}")
    print(f"  Average Days Up:        {analysis['avg_days_up']:.1f}")
    print(f"  Gain Range:             {analysis['gain_range'][0]:.1f}% - {analysis['gain_range'][1]:.1f}%")
    
    print(f"\n  Top 10 Most Frequent Symbols:")
    for symbol, count in analysis['top_symbols']:
        print(f"    {symbol:<10} {count:>3} setups")
    
    print(f"\n{'='*80}")
    
    # Export to CSV
    csv_path = historical_screener.export_setups_for_backtest(setups)
    print(f"\nAll setups exported to: {csv_path}")


def test_single_setup(symbol: str, date_str: str):
    """Test a single known parabolic setup."""
    print_banner()
    
    date = datetime.strptime(date_str, '%Y-%m-%d')
    
    print(f"Testing single setup:")
    print(f"  Symbol: {symbol}")
    print(f"  Date: {date_str}")
    print()
    
    result = tick_backtest_engine.run_tick_backtest(
        symbol=symbol,
        date=date,
        verbose=True
    )
    
    # Generate HTML report
    from src.backtest.visualizer import visualizer
    report_path = visualizer.generate_html_report(result)
    print(f"\nDetailed report: {report_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Historical Parabolic Reversal Backtest'
    )
    
    parser.add_argument('--full', action='store_true',
                       help='Run full 2019-2024 backtest (takes time)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test with 10 setups')
    parser.add_argument('--scan', action='store_true',
                       help='Scan and list all historical setups')
    parser.add_argument('--symbol', type=str,
                       help='Test single symbol')
    parser.add_argument('--date', type=str,
                       help='Date for single test (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    if args.full:
        run_full_backtest()
    elif args.quick_test:
        run_quick_test()
    elif args.scan:
        scan_and_list_setups()
    elif args.symbol and args.date:
        test_single_setup(args.symbol, args.date)
    else:
        print_banner()
        print("Usage:")
        print("  python run_historical_backtest.py --full")
        print("  python run_historical_backtest.py --quick-test")
        print("  python run_historical_backtest.py --scan")
        print("  python run_historical_backtest.py --symbol TSLA --date 2023-01-15")
        print()


if __name__ == "__main__":
    main()
