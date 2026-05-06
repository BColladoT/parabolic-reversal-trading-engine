"""Test V3 engine."""
from datetime import datetime
from src.backtest.tick_backtest_engine_v3 import tick_backtest_engine_v3

setups = [
    ('WWR', '2020-10-05'),
    ('AMC', '2021-06-02'),
    ('RENT', '2024-04-11'),
]

print("\n" + "="*70)
print("TESTING V3 ENGINE (2-of-3 Criteria)")
print("="*70)

for symbol, date_str in setups:
    date = datetime.strptime(date_str, '%Y-%m-%d')
    result = tick_backtest_engine_v3.run_tick_backtest(symbol, date, verbose=True)
    print(f"\nResult: {result.total_trades} trades, ${result.total_pnl:+.2f}")
    print("-" * 70)
