"""
Backtest all 909 relaxed setups with V5 engine
"""
import sys
from pathlib import Path
from datetime import datetime
import pickle
import pandas as pd

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.backtest.tick_backtest_engine_v5 import TickBacktestEngineV5

# Load setups
cache_path = Path('data/cache/setups/setups_relaxed_full_2019_2024.pkl')
with open(cache_path, 'rb') as f:
    setups = pickle.load(f)

print("="*80)
print(f"BACKTESTING {len(setups)} RELAXED SETUPS WITH V5 ENGINE")
print("="*80)
print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
print()

engine = TickBacktestEngineV5()
results = []

for i, setup in enumerate(setups):
    if i % 20 == 0:
        print(f"Progress: {i}/{len(setups)} setups ({i/len(setups)*100:.0f}%)")
    
    try:
        result = engine.run_tick_backtest(setup.symbol, setup.date, verbose=False)
        results.append({
            'symbol': setup.symbol,
            'date': setup.date.strftime('%Y-%m-%d'),
            'gain_pct': setup.gain_percent,
            'trades': result.total_trades,
            'pnl': result.total_pnl,
            'win': 1 if result.total_pnl > 0 else 0,
            'loss': 1 if result.total_pnl <= 0 else 0
        })
    except Exception as e:
        print(f"  Error on {setup.symbol}: {e}")
        continue

print(f"\n[{len(results)}/{len(setups)} setups processed successfully]")

# Analysis
df = pd.DataFrame(results)

total_setups = len(df)
setups_with_trades = len(df[df['trades'] > 0])
conversion_rate = setups_with_trades / total_setups * 100

total_trades = df['trades'].sum()
total_pnl = df['pnl'].sum()
winning_trades = df['win'].sum()
losing_trades = df['loss'].sum()
win_rate = winning_trades / setups_with_trades * 100 if setups_with_trades > 0 else 0

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Setups tested:       {total_setups}")
print(f"Setups with trades:  {setups_with_trades}")
print(f"Conversion rate:     {conversion_rate:.1f}%")
print(f"")
print(f"Total trades:        {int(total_trades)}")
print(f"Winning trades:      {winning_trades}")
print(f"Losing trades:       {losing_trades}")
print(f"Win rate:            {win_rate:.1f}%")
print(f"")
print(f"Total P&L:           ${total_pnl:+.2f}")
if total_trades > 0:
    print(f"Avg P&L per trade:   ${total_pnl/total_trades:.2f}")

# Top winners
print(f"\n[TOP 15 WINNERS]")
top = df.nlargest(15, 'pnl')
for _, row in top.iterrows():
    print(f"  {row['symbol']:6} {row['date']} | ${row['pnl']:+8.2f} | {row['gain_pct']:.1f}% gap")

# Top losers
print(f"\n[TOP 10 LOSERS]")
bottom = df.nsmallest(10, 'pnl')
for _, row in bottom.iterrows():
    print(f"  {row['symbol']:6} {row['date']} | ${row['pnl']:+8.2f} | {row['gain_pct']:.1f}% gap")

# Comparison
print("\n" + "="*80)
print("COMPARISON: Original vs Relaxed")
print("="*80)
print(f"{'Metric':<20} {'Original':<15} {'Relaxed':<15}")
print("-" * 50)
print(f"{'Setups':<20} {'242':<15} {str(total_setups):<15}")
print(f"{'Trades':<20} {'40':<15} {str(int(total_trades)):<15}")
print(f"{'Win Rate':<20} {'80.0%':<15} {f'{win_rate:.1f}%':<15}")
print(f"{'Total P&L':<20} {'$+53,148':<15} {f'${total_pnl:+.0f}':<15}")
print(f"{'Trades/Year':<20} {'~7':<15} {f'~{int(total_trades/6)}':<15}")

# Save
print(f"\nEnd time: {datetime.now().strftime('%H:%M:%S')}")
df.to_csv("reports/relaxed_909_backtest.csv", index=False)
print("\nSaved: reports/relaxed_909_backtest.csv")
