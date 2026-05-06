"""
Real-time Backtest Monitor Dashboard

Monitors the progress of a running backtest and displays live statistics.

Usage:
    python monitor_backtest.py
    
Press Ctrl+C to exit (monitor runs until backtest completes)
"""

import time
import json
import curses
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd


def format_money(value: float) -> str:
    """Format money value."""
    if abs(value) >= 1000000:
        return f"${value/1000000:.2f}M"
    elif abs(value) >= 1000:
        return f"${value/1000:.1f}K"
    else:
        return f"${value:.0f}"


def load_data() -> dict:
    """Load current backtest data."""
    data = {
        'checkpoint': None,
        'summary': None
    }
    
    # Try to load checkpoint
    checkpoint_path = Path('reports/comparison_checkpoint.csv')
    if checkpoint_path.exists():
        try:
            data['checkpoint'] = pd.read_csv(checkpoint_path)
        except:
            pass
    
    # Try to load summary
    summary_path = Path('reports/comparison_summary.json')
    if summary_path.exists():
        try:
            with open(summary_path, 'r') as f:
                data['summary'] = json.load(f)
        except:
            pass
    
    return data


def draw_dashboard(stdscr):
    """Draw the monitoring dashboard."""
    # Setup
    curses.curs_set(0)  # Hide cursor
    stdscr.nodelay(1)   # Non-blocking input
    
    while True:
        # Clear screen
        stdscr.clear()
        
        # Load data
        data = load_data()
        
        # Header
        stdscr.addstr(0, 0, "=" * 80, curses.A_BOLD)
        stdscr.addstr(1, 0, " BACKTEST COMPARISON MONITOR ", curses.A_BOLD | curses.A_REVERSE)
        stdscr.addstr(1, 60, f" {datetime.now().strftime('%H:%M:%S')} ")
        stdscr.addstr(2, 0, "=" * 80, curses.A_BOLD)
        
        if data['checkpoint'] is None:
            stdscr.addstr(4, 0, "Waiting for backtest to start...")
            stdscr.addstr(5, 0, "(Run: python run_full_comparison_backtest.py)")
            stdscr.refresh()
            time.sleep(1)
            continue
        
        df = data['checkpoint']
        
        # Calculate stats
        v5_df = df[df['strategy'] == 'v5_relaxed']
        inst_df = df[df['strategy'] == 'v5_institutional']
        
        v5_trades = v5_df[v5_df['pnl'] != 0]
        inst_trades = inst_df[inst_df['pnl'] != 0]
        inst_blocked = len(inst_df[inst_df['pnl'] == 0])
        
        # Progress
        total_processed = len(df) // 2  # Each setup generates 2 records
        stdscr.addstr(4, 0, f"Progress: {total_processed} setups processed")
        
        # V5 Stats
        row = 6
        stdscr.addstr(row, 0, "V5 RELAXED SCANNER", curses.A_BOLD | curses.COLOR_GREEN)
        row += 1
        stdscr.addstr(row, 2, f"Trades: {len(v5_trades)}")
        row += 1
        if len(v5_trades) > 0:
            win_rate = v5_trades['win'].mean() * 100
            total_pnl = v5_trades['pnl'].sum()
            avg_trade = v5_trades['pnl'].mean()
            stdscr.addstr(row, 2, f"Win Rate: {win_rate:.1f}%")
            row += 1
            stdscr.addstr(row, 2, f"Total P&L: {format_money(total_pnl)}")
            row += 1
            stdscr.addstr(row, 2, f"Avg Trade: {format_money(avg_trade)}")
            row += 2
        
        # Institutional Stats
        stdscr.addstr(row, 0, "V5 INSTITUTIONAL ML", curses.A_BOLD | curses.COLOR_BLUE)
        row += 1
        stdscr.addstr(row, 2, f"Trades: {len(inst_trades)} | Blocked: {inst_blocked}")
        row += 1
        if len(inst_trades) > 0:
            win_rate = inst_trades['win'].mean() * 100
            total_pnl = inst_trades['pnl'].sum()
            avg_trade = inst_trades['pnl'].mean()
            block_rate = inst_blocked / len(inst_df) * 100 if len(inst_df) > 0 else 0
            stdscr.addstr(row, 2, f"Win Rate: {win_rate:.1f}%")
            row += 1
            stdscr.addstr(row, 2, f"Total P&L: {format_money(total_pnl)}")
            row += 1
            stdscr.addstr(row, 2, f"Avg Trade: {format_money(avg_trade)}")
            row += 1
            stdscr.addstr(row, 2, f"Block Rate: {block_rate:.1f}%")
            row += 2
        
        # Comparison
        if len(v5_trades) > 0 and len(inst_trades) > 0:
            stdscr.addstr(row, 0, "COMPARISON", curses.A_BOLD)
            row += 1
            pnl_diff = inst_trades['pnl'].sum() - v5_trades['pnl'].sum()
            wr_diff = (inst_trades['win'].mean() - v5_trades['win'].mean()) * 100
            stdscr.addstr(row, 2, f"P&L Diff: {format_money(pnl_diff)}")
            row += 1
            stdscr.addstr(row, 2, f"Win Rate Diff: {wr_diff:+.1f}%")
            row += 2
        
        # Recent trades
        stdscr.addstr(row, 0, "RECENT TRADES", curses.A_BOLD)
        row += 1
        recent = df.tail(10)
        for _, trade in recent.iterrows():
            if row >= 35:
                break
            symbol = trade['symbol']
            pnl = trade['pnl']
            strategy = "V5" if trade['strategy'] == 'v5_relaxed' else "ML"
            color = curses.COLOR_GREEN if pnl > 0 else curses.COLOR_RED if pnl < 0 else curses.COLOR_WHITE
            stdscr.addstr(row, 2, f"{symbol:8s} {strategy:3s} {format_money(pnl):>10s}", color)
            row += 1
        
        # Footer
        stdscr.addstr(37, 0, "=" * 80, curses.A_BOLD)
        stdscr.addstr(38, 0, "Press Ctrl+C to exit monitor")
        
        stdscr.refresh()
        
        # Check for exit
        try:
            key = stdscr.getch()
            if key == ord('q'):
                break
        except:
            pass
        
        time.sleep(1)


def simple_monitor():
    """Simple text-based monitor (no curses)."""
    print("="*80)
    print("BACKTEST COMPARISON MONITOR (Text Mode)")
    print("="*80)
    print("\nMonitoring... Press Ctrl+C to exit\n")
    
    try:
        while True:
            data = load_data()
            
            if data['checkpoint'] is None:
                print("Waiting for backtest to start...")
                time.sleep(2)
                continue
            
            df = data['checkpoint']
            
            # Clear screen (cross-platform)
            print("\033[2J\033[H")
            
            print("="*80)
            print(f"BACKTEST MONITOR - {datetime.now().strftime('%H:%M:%S')}")
            print("="*80)
            
            # Calculate stats
            v5_df = df[df['strategy'] == 'v5_relaxed']
            inst_df = df[df['strategy'] == 'v5_institutional']
            
            v5_trades = v5_df[v5_df['pnl'] != 0]
            inst_trades = inst_df[inst_df['pnl'] != 0]
            inst_blocked = len(inst_df[inst_df['pnl'] == 0])
            
            total = len(df) // 2
            
            print(f"\nProcessed: {total} setups")
            
            print("\n--- V5 RELAXED ---")
            print(f"  Trades: {len(v5_trades)}")
            if len(v5_trades) > 0:
                print(f"  Win Rate: {v5_trades['win'].mean()*100:.1f}%")
                print(f"  Total P&L: ${v5_trades['pnl'].sum():,.0f}")
                print(f"  Avg Trade: ${v5_trades['pnl'].mean():,.0f}")
            
            print("\n--- V5 INSTITUTIONAL ---")
            print(f"  Trades: {len(inst_trades)} | Blocked: {inst_blocked}")
            if len(inst_trades) > 0:
                print(f"  Win Rate: {inst_trades['win'].mean()*100:.1f}%")
                print(f"  Total P&L: ${inst_trades['pnl'].sum():,.0f}")
                print(f"  Avg Trade: ${inst_trades['pnl'].mean():,.0f}")
                print(f"  Block Rate: {inst_blocked/len(inst_df)*100:.1f}%" if len(inst_df) > 0 else "  Block Rate: N/A")
            
            if len(v5_trades) > 0 and len(inst_trades) > 0:
                print("\n--- COMPARISON ---")
                pnl_diff = inst_trades['pnl'].sum() - v5_trades['pnl'].sum()
                print(f"  P&L Difference: ${pnl_diff:+,.0f}")
                wr_diff = (inst_trades['win'].mean() - v5_trades['win'].mean()) * 100
                print(f"  Win Rate Difference: {wr_diff:+.1f}%")
            
            print("\n--- RECENT TRADES ---")
            recent = df.tail(6)
            for _, trade in recent.iterrows():
                symbol = trade['symbol']
                pnl = trade['pnl']
                strategy = "V5" if trade['strategy'] == 'v5_relaxed' else "ML"
                status = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "SKIP"
                print(f"  {symbol:8s} {strategy:3s} {status:6s} ${pnl:>10,.0f}")
            
            print("\n" + "="*80)
            print("Refreshing in 5 seconds... (Ctrl+C to exit)")
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\nMonitor stopped.")


if __name__ == "__main__":
    # Try curses first, fall back to simple monitor
    try:
        import curses
        curses.wrapper(draw_dashboard)
    except:
        simple_monitor()
