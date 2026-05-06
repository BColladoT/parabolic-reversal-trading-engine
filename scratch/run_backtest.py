#!/usr/bin/env python3
"""
Parabolic Reversal Strategy - Backtest Runner
Run backtests with full audit trails and reasoning.

Usage:
    python run_backtest.py --symbol AAPL --date 2024-01-15
    python run_backtest.py --find-candidates --days 30
    python run_backtest.py --batch --symbols AAPL TSLA NVDA
"""
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from alpaca.data.timeframe import TimeFrame
import polars as pl

from src.backtest.backtest_engine import backtest_engine
from src.backtest.data_fetcher import data_fetcher
from src.backtest.visualizer import visualizer
from src.utils.logger import logger


def find_candidates(args):
    """Find parabolic candidates for backtesting."""
    print(f"\n🔍 Scanning for parabolic moves in last {args.days} days...")
    
    # Common low-float / high-volatility symbols
    symbols = [
        'AAPL', 'TSLA', 'NVDA', 'AMD', 'META', 'AMZN', 'MSFT', 'GOOGL',
        'NFLX', 'CRM', 'UBER', 'COIN', 'PLTR', 'HOOD', 'SOFI', 'RBLX'
    ]
    
    if args.symbols:
        symbols = args.symbols
    
    candidates = data_fetcher.find_parabolic_candidates(
        symbols=symbols,
        lookback_days=args.days,
        min_gain_percent=args.min_gain
    )
    
    print(f"\n✅ Found {len(candidates)} parabolic candidates:\n")
    print(f"{'Symbol':<10} {'Date':<12} {'Gain':<10} {'Open':<10} {'High':<10} {'Close':<10}")
    print("-" * 70)
    
    for c in candidates[:20]:  # Show top 20
        date_str = c['date'].strftime('%Y-%m-%d') if isinstance(c['date'], datetime) else str(c['date'])[:10]
        print(f"{c['symbol']:<10} {date_str:<12} {c['gain_percent']:>7.1f}%  "
              f"${c['open']:<9.2f} ${c['high']:<9.2f} ${c['close']:<9.2f}")
    
    # Save to file
    output_file = Path("reports/parabolic_candidates.txt")
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(f"Parabolic Candidates (Last {args.days} days, Min {args.min_gain}% gain)\n")
        f.write("=" * 70 + "\n\n")
        for c in candidates:
            date_str = c['date'].strftime('%Y-%m-%d') if isinstance(c['date'], datetime) else str(c['date'])[:10]
            f.write(f"{c['symbol']} | {date_str} | +{c['gain_percent']:.1f}% | "
                   f"O:${c['open']:.2f} H:${c['high']:.2f} C:${c['close']:.2f}\n")
    
    print(f"\n📁 Saved to {output_file}")
    return candidates


def run_single_backtest(args):
    """Run backtest for a single symbol/date."""
    date = datetime.strptime(args.date, '%Y-%m-%d')
    
    print(f"\n{'='*60}")
    print(f"BACKTEST: {args.symbol} on {args.date}")
    print(f"{'='*60}\n")
    
    # Run backtest
    result = backtest_engine.run_backtest(
        symbol=args.symbol,
        date=date,
        verbose=True
    )
    
    if not result.audit_records:
        print("\n⚠️  No trades executed on this date")
        return
    
    # Generate visualizations
    if not args.no_charts:
        print("\n📊 Generating charts...")
        
        # Equity curve
        chart_path = f"reports/{args.symbol}_{args.date}_equity.png"
        visualizer.create_equity_curve(result, save_path=chart_path)
        
        # Price chart with trades
        df = data_fetcher.get_intraday_for_date(args.symbol, date)
        if not df.is_empty():
            import pandas as pd
            price_data = df.to_pandas()
            price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
            price_data.set_index('timestamp', inplace=True)
            
            chart_path = f"reports/{args.symbol}_{args.date}_trades.png"
            visualizer.create_trade_chart(result, price_data, save_path=chart_path)
    
    # Generate HTML report
    if args.html:
        report_path = visualizer.generate_html_report(result)
        print(f"\n📄 HTML Report: {report_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total Trades:    {result.total_trades}")
    print(f"Win Rate:        {result.win_rate:.1%}")
    print(f"Total P&L:       ${result.total_pnl:+.2f}")
    print(f"Profit Factor:   {result.profit_factor:.2f}")
    print(f"{'='*60}\n")


def run_batch_backtest(args):
    """Run backtests for multiple symbols."""
    symbols = args.symbols
    
    # Default date range (last 30 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    print(f"\n📊 Running batch backtest for {len(symbols)} symbols...")
    print(f"Period: {start_date.date()} to {end_date.date()}\n")
    
    all_results = []
    
    for symbol in symbols:
        # Find parabolic days for this symbol
        df = data_fetcher.fetch_alpaca_bars(
            symbol=symbol,
            start=start_date,
            end=end_date,
            timeframe=TimeFrame.Day
        )
        
        if df.is_empty():
            continue
        
        # Find days with >50% gains
        df = df.with_columns([
            (((pl.col('close') / pl.col('open')) - 1) * 100).alias('gain_pct')
        ])
        
        parabolic_days = df.filter(pl.col('gain_pct') >= 50.0)
        
        for row in parabolic_days.to_dicts():
            date = row['timestamp']
            print(f"  Testing {symbol} on {date.date()} (+{row['gain_pct']:.1f}%)...")
            
            result = backtest_engine.run_backtest(
                symbol=symbol,
                date=date,
                verbose=False
            )
            all_results.append(result)
    
    # Aggregate results
    total_pnl = sum(r.total_pnl for r in all_results)
    total_trades = sum(r.total_trades for r in all_results)
    winning_trades = sum(r.winning_trades for r in all_results)
    
    print(f"\n{'='*60}")
    print(f"BATCH BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"Symbols tested:  {len(symbols)}")
    print(f"Total setups:    {len(all_results)}")
    print(f"Total trades:    {total_trades}")
    print(f"Win rate:        {winning_trades/total_trades:.1%}" if total_trades > 0 else "N/A")
    print(f"Total P&L:       ${total_pnl:+.2f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Parabolic Reversal Strategy Backtester'
    )
    
    parser.add_argument('--find-candidates', action='store_true',
                       help='Find parabolic candidates for backtesting')
    parser.add_argument('--symbol', type=str,
                       help='Symbol to backtest')
    parser.add_argument('--date', type=str,
                       help='Date to backtest (YYYY-MM-DD)')
    parser.add_argument('--days', type=int, default=30,
                       help='Number of days to look back (default: 30)')
    parser.add_argument('--min-gain', type=float, default=80.0,
                       help='Minimum gain % for candidates (default: 80)')
    parser.add_argument('--batch', action='store_true',
                       help='Run batch backtest on multiple symbols')
    parser.add_argument('--symbols', nargs='+',
                       help='List of symbols for batch backtest')
    parser.add_argument('--no-charts', action='store_true',
                       help='Skip chart generation')
    parser.add_argument('--html', action='store_true',
                       help='Generate HTML report')
    
    args = parser.parse_args()
    
    if not any([args.find_candidates, args.symbol, args.batch]):
        parser.print_help()
        return
    
    try:
        if args.find_candidates:
            find_candidates(args)
        elif args.batch:
            if not args.symbols:
                print("Error: --batch requires --symbols")
                return
            run_batch_backtest(args)
        elif args.symbol and args.date:
            run_single_backtest(args)
        else:
            print("Error: Need both --symbol and --date for single backtest")
            
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise


if __name__ == "__main__":
    main()
