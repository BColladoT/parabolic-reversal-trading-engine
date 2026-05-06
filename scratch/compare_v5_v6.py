"""
Compare V5 vs V6 Strategy Performance
Tests both engines on the same setups to see trade frequency vs win rate trade-off.
"""
import sys
from pathlib import Path
from datetime import datetime
import pickle

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.backtest.tick_backtest_engine_v5 import TickBacktestEngineV5
from src.backtest.tick_backtest_engine_v6 import TickBacktestEngineV6
from src.backtest.historical_screener import historical_screener


# Test on top performing symbols/dates
TEST_SETUPS = [
    ('GCT', '2022-08-19'),
    ('GDC', '2024-08-21'),
    ('RENT', '2024-04-11'),
    ('CRBP', '2024-01-26'),
    ('LIDR', '2024-05-10'),
    ('WWR', '2020-10-05'),
    ('HOUR', '2024-12-24'),
    ('WKEY', '2024-12-13'),
    ('OCGN', '2021-02-08'),
    ('VANI', '2021-03-05'),
    ('MNOV', '2020-07-27'),
    ('SPRU', '2020-12-23'),
    ('MPU', '2020-12-28'),
    ('SENS', '2021-01-19'),
    ('OCGN', '2021-02-08'),
]


def test_strategy(symbol: str, date_str: str):
    """Test both V5 and V6 on same setup."""
    date = datetime.strptime(date_str, "%Y-%m-%d")
    
    print(f"\n{'='*60}")
    print(f"Testing {symbol} on {date_str}")
    print(f"{'='*60}")
    
    # V5 Test
    print("\n[V5 - Conservative]")
    v5 = TickBacktestEngineV5()
    try:
        result_v5 = v5.run_tick_backtest(symbol, date, verbose=False)
        v5_trades = result_v5.total_trades
        v5_pnl = result_v5.total_pnl
    except Exception as e:
        print(f"  Error: {e}")
        v5_trades = 0
        v5_pnl = 0
    
    # V6 Test
    print("\n[V6 - High Frequency]")
    v6 = TickBacktestEngineV6()
    try:
        result_v6 = v6.run_tick_backtest(symbol, date, verbose=False)
        v6_trades = result_v6.total_trades
        v6_pnl = result_v6.total_pnl
    except Exception as e:
        print(f"  Error: {e}")
        v6_trades = 0
        v6_pnl = 0
    
    print(f"\n[RESULTS]")
    print(f"  V5: {v5_trades} trades, P&L: ${v5_pnl:+.2f}")
    print(f"  V6: {v6_trades} trades, P&L: ${v6_pnl:+.2f}")
    
    if v6_trades > v5_trades:
        print(f"  => V6 captured {v6_trades - v5_trades} additional trade(s)")
    elif v5_trades > 0 and v6_trades == 0:
        print(f"  => V6 MISSED the trade V5 caught")
    else:
        print(f"  => Same trade count")
    
    return {
        'symbol': symbol,
        'date': date_str,
        'v5_trades': v5_trades,
        'v5_pnl': v5_pnl,
        'v6_trades': v6_trades,
        'v6_pnl': v6_pnl
    }


def main():
    print("="*70)
    print("STRATEGY COMPARISON: V5 (Conservative) vs V6 (High Frequency)")
    print("="*70)
    print("\nV5 Criteria:")
    print("  - Entry: 2-of-3 (VWAP>1.15x, Vol<70%, Prox>93%)")
    print("  - Min gain: 50%, Must be above VWAP")
    print("  - Window: 9:45-2:00, Max 1 trade/day")
    print("\nV6 Criteria:")
    print("  - Entry: 1-of-3 (VWAP>1.12x, Vol<80%, Prox>85%)")
    print("  - Min gain: 40%, Can be below VWAP with bonus")
    print("  - Window: 9:35-2:30, Max 3 trades/day")
    
    results = []
    for symbol, date in TEST_SETUPS:
        try:
            result = test_strategy(symbol, date)
            results.append(result)
        except Exception as e:
            print(f"\nError on {symbol} {date}: {e}")
            continue
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    total_v5_trades = sum(r['v5_trades'] for r in results)
    total_v6_trades = sum(r['v6_trades'] for r in results)
    total_v5_pnl = sum(r['v5_pnl'] for r in results)
    total_v6_pnl = sum(r['v6_pnl'] for r in results)
    
    print(f"\nTotal Trades:")
    print(f"  V5: {total_v5_trades} trades")
    print(f"  V6: {total_v6_trades} trades ({total_v6_trades - total_v5_trades:+d} vs V5)")
    
    print(f"\nTotal P&L:")
    print(f"  V5: ${total_v5_pnl:+.2f}")
    print(f"  V6: ${total_v6_pnl:+.2f} ({total_v6_pnl - total_v5_pnl:+.2f} vs V5)")
    
    if total_v5_trades > 0:
        avg_v5 = total_v5_pnl / total_v5_trades
        print(f"\nAvg P&L per Trade:")
        print(f"  V5: ${avg_v5:.2f}")
    
    if total_v6_trades > 0:
        avg_v6 = total_v6_pnl / total_v6_trades
        print(f"  V6: ${avg_v6:.2f}")
    
    # Calculate win rates
    v5_wins = sum(1 for r in results if r['v5_pnl'] > 0)
    v6_wins = sum(1 for r in results if r['v6_pnl'] > 0)
    v5_tested = sum(1 for r in results if r['v5_trades'] > 0)
    v6_tested = sum(1 for r in results if r['v6_trades'] > 0)
    
    if v5_tested > 0:
        print(f"\nWin Rate (tested setups with trades):")
        print(f"  V5: {v5_wins}/{v5_tested} = {v5_wins/v5_tested*100:.1f}%")
    if v6_tested > 0:
        print(f"  V6: {v6_wins}/{v6_tested} = {v6_wins/v6_tested*100:.1f}%")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
