"""Test the V2 backtest engine on multiple setups."""
from datetime import datetime
from src.backtest.tick_backtest_engine_v2 import tick_backtest_engine_v2

def test_setup(symbol, date_str):
    date = datetime.strptime(date_str, '%Y-%m-%d')
    result = tick_backtest_engine_v2.run_tick_backtest(symbol, date, verbose=True)
    return result

if __name__ == "__main__":
    # Test multiple setups
    setups = [
        ('WWR', '2020-10-05'),
        ('AMC', '2021-06-02'),
        ('RENT', '2024-04-11'),
        ('GME', '2021-01-27'),
    ]
    
    results = []
    for symbol, date in setups:
        result = test_setup(symbol, date)
        results.append({
            'symbol': symbol,
            'date': date,
            'trades': result.total_trades,
            'pnl': result.total_pnl
        })
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for r in results:
        status = "✓" if r['trades'] > 0 else "✗"
        print(f"{status} {r['symbol']} {r['date']}: {r['trades']} trades, ${r['pnl']:+.2f}")
