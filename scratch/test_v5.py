"""Test V5 engine - Hybrid."""
from datetime import datetime
from src.backtest.tick_backtest_engine_v5 import tick_backtest_engine_v5

setups = [
    ('WWR', '2020-10-05'),
    ('AMC', '2021-06-02'),
    ('RENT', '2024-04-11'),
    ('GME', '2021-01-27'),
]

print("\n" + "="*70)
print("TESTING V5 ENGINE (Hybrid - Best of V3 + V4)")
print("="*70)

total_pnl = 0
total_trades = 0

for symbol, date_str in setups:
    date = datetime.strptime(date_str, '%Y-%m-%d')
    result = tick_backtest_engine_v5.run_tick_backtest(symbol, date, verbose=True)
    
    total_pnl += result.total_pnl
    total_trades += result.total_trades
    
    print("-" * 70)

print("\n" + "="*70)
print(f"TOTAL: {total_trades} trades, ${total_pnl:+.2f}")
print("="*70)
