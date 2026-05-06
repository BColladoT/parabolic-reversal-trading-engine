"""
Monitor for Fresh Complete Backtest

Real-time monitoring dashboard for the full 6-year fresh scan.
"""

import time
import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd


def clear_screen():
    """Clear terminal screen."""
    print("\033[2J\033[H")


def format_money(value: float) -> str:
    """Format money value."""
    if abs(value) >= 1000000:
        return f"${value/1000000:.2f}M"
    elif abs(value) >= 1000:
        return f"${value/1000:.1f}K"
    else:
        return f"${value:.0f}"


def load_data():
    """Load current backtest data."""
    data = {'trades': None, 'stats': None}
    
    output_dir = Path('reports/complete_fresh_backtest')
    
    checkpoint_path = output_dir / 'checkpoint.csv'
    if checkpoint_path.exists():
        try:
            data['trades'] = pd.read_csv(checkpoint_path)
        except:
            pass
    
    stats_path = output_dir / 'stats.json'
    if stats_path.exists():
        try:
            with open(stats_path) as f:
                data['stats'] = json.load(f)
        except:
            pass
    
    return data


def main():
    print("="*80)
    print("FRESH BACKTEST MONITOR")
    print("Monitoring complete 6-year scan with ML risk engine")
    print("="*80)
    print("\nWaiting for backtest to start...")
    
    try:
        while True:
            data = load_data()
            
            if data['stats'] is None:
                print(".", end='', flush=True)
                time.sleep(2)
                continue
            
            clear_screen()
            
            print("="*80)
            print(f"FRESH BACKTEST MONITOR - {datetime.now().strftime('%H:%M:%S')}")
            print("="*80)
            
            stats = data['stats']
            
            # Progress
            symbols_scanned = stats.get('symbols_scanned', 0)
            total_symbols = 3527
            pct = (symbols_scanned / total_symbols) * 100 if total_symbols > 0 else 0
            
            print(f"\nProgress: {symbols_scanned}/{total_symbols} symbols ({pct:.1f}%)")
            print(f"Days Processed: {stats.get('days_processed', 0)}")
            print(f"Current: {stats.get('current_symbol', 'N/A')} on {stats.get('current_date', 'N/A')}")
            
            # V5 Stats
            print("\n" + "-"*80)
            print("V5 RELAXED SCANNER")
            print("-"*80)
            v5_trades = stats.get('v5_trades_taken', 0)
            v5_pnl = stats.get('v5_pnl', 0)
            print(f"  Trades Taken: {v5_trades}")
            print(f"  Total P&L: {format_money(v5_pnl)}")
            if v5_trades > 0:
                print(f"  Average: {format_money(v5_pnl/v5_trades)}")
            
            # ML Stats
            print("\n" + "-"*80)
            print("V5 INSTITUTIONAL ML")
            print("-"*80)
            ml_trades = stats.get('ml_trades_taken', 0)
            ml_blocked = stats.get('ml_trades_blocked', 0)
            ml_pnl = stats.get('ml_pnl', 0)
            total_ml = ml_trades + ml_blocked
            block_rate = (ml_blocked / total_ml * 100) if total_ml > 0 else 0
            
            print(f"  Trades Taken: {ml_trades}")
            print(f"  Trades Blocked: {ml_blocked} ({block_rate:.1f}%)")
            print(f"  Total P&L: {format_money(ml_pnl)}")
            if ml_trades > 0:
                print(f"  Average: {format_money(ml_pnl/ml_trades)}")
            
            # Comparison
            print("\n" + "-"*80)
            print("COMPARISON")
            print("-"*80)
            pnl_diff = ml_pnl - v5_pnl
            print(f"  P&L Difference: {format_money(pnl_diff)} ({'ML ahead' if pnl_diff > 0 else 'V5 ahead'})")
            print(f"  Trades Avoided: {ml_blocked}")
            
            # Recent trades
            if data['trades'] is not None and len(data['trades']) > 0:
                print("\n" + "-"*80)
                print("RECENT SETUPS")
                print("-"*80)
                recent = data['trades'].tail(10)
                for _, trade in recent.iterrows():
                    symbol = trade['symbol']
                    strat = trade['strategy']
                    pnl = trade['pnl']
                    blocked = trade.get('ml_blocked', False)
                    
                    if strat == 'v5_relaxed':
                        status = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "SKIP"
                        print(f"  {symbol:8s} V5   {status:6s} {format_money(pnl):>10s}")
                    else:
                        if blocked:
                            print(f"  {symbol:8s} ML   BLOCKED")
                        else:
                            status = "WIN" if pnl > 0 else "LOSS"
                            print(f"  {symbol:8s} ML   {status:6s} {format_money(pnl):>10s}")
            
            print("\n" + "="*80)
            print("Refreshing in 10 seconds... (Ctrl+C to exit)")
            print("="*80)
            
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n\nMonitor stopped.")
        print("\nTo view final results:")
        print("  cat reports/complete_fresh_backtest/report.json")


if __name__ == "__main__":
    main()
