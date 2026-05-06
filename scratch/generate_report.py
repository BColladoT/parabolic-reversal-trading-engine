"""Generate comprehensive report from relaxed backtest results"""
import pandas as pd
from pathlib import Path
from datetime import datetime

# Load results
df = pd.read_csv('reports/relaxed_909_backtest.csv')

print('='*80)
print('RELAXED 909 SETUPS BACKTEST - FINAL RESULTS')
print('='*80)
print(f'Analysis time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print()

# Metrics
total_setups = len(df)
setups_with_trades = len(df[df['trades'] > 0])
conversion_rate = setups_with_trades / total_setups * 100 if total_setups > 0 else 0
total_trades = df['trades'].sum()
total_pnl = df['pnl'].sum()
winning = len(df[df['pnl'] > 0])
losing = len(df[df['pnl'] < 0])
win_rate = winning / setups_with_trades * 100 if setups_with_trades > 0 else 0

print('[OVERALL METRICS]')
print(f'Total setups tested:     {total_setups}')
print(f'Setups with trades:      {setups_with_trades}')
print(f'Conversion rate:         {conversion_rate:.1f}%')
print()
print(f'Total trades executed:   {int(total_trades)}')
print(f'Winning trades:          {winning}')
print(f'Losing trades:           {losing}')
print(f'Win rate:                {win_rate:.1f}%')
print()
print(f'Total PnL:               ${total_pnl:+.2f}')
if total_trades > 0:
    print(f'Avg PnL per trade:       ${total_pnl/total_trades:.2f}')
print()

# Top 20 winners
print('='*80)
print('TOP 20 WINNING TRADES')
print('='*80)
top = df.nlargest(20, 'pnl')
for i, (_, row) in enumerate(top.iterrows(), 1):
    print(f"{i:2}. {row['symbol']:6} {row['date']} | ${row['pnl']:+9.2f} | {row['gain_pct']:5.1f}% gap")

# Top 10 losers
print()
print('='*80)
print('TOP 10 LOSING TRADES')
print('='*80)
bottom = df.nsmallest(10, 'pnl')
for i, (_, row) in enumerate(bottom.iterrows(), 1):
    print(f"{i:2}. {row['symbol']:6} {row['date']} | ${row['pnl']:+9.2f} | {row['gain_pct']:5.1f}% gap")

# Monthly analysis
print()
print('='*80)
print('MONTHLY BREAKDOWN (Last 15 months with trades)')
print('='*80)
df['date'] = pd.to_datetime(df['date'])
df['year_month'] = df['date'].dt.to_period('M')
monthly = df.groupby('year_month').agg({
    'trades': 'sum',
    'pnl': 'sum'
}).reset_index()
monthly = monthly[monthly['trades'] > 0].sort_values('year_month')
print(f"{'Month':12} {'Trades':>8} {'PnL':>12}")
print('-' * 35)
for _, row in monthly.tail(15).iterrows():
    print(f"{str(row['year_month']):12} {int(row['trades']):>8} ${row['pnl']:>+10.2f}")

# Comparison
print()
print('='*80)
print('COMPARISON: Original vs Relaxed Criteria')
print('='*80)
orig_pnl = 53148.33
orig_trades = 40
orig_winrate = 80.0
print(f"{'Metric':<25} {'Original (50%)':<18} {'Relaxed (30%)':<18}")
print('-' * 62)
print(f"{'Setups Found':<25} {'242':<18} {str(total_setups):<18}")
print(f"{'Trades Taken':<25} {str(orig_trades):<18} {str(int(total_trades)):<18} ({(total_trades/orig_trades-1)*100:+.0f}%)")
print(f"{'Win Rate':<25} {f'{orig_winrate:.1f}%':<18} {f'{win_rate:.1f}%':<18}")
print(f"{'Total PnL':<25} {f'${orig_pnl:,.0f}':<18} {f'${total_pnl:,.0f}':<18} ({(total_pnl/orig_pnl-1)*100:+.0f}%)")
avg_orig = orig_pnl/orig_trades
avg_new = total_pnl/total_trades if total_trades else 0
print(f"{'Avg per Trade':<25} {f'${avg_orig:,.0f}':<18} {f'${avg_new:,.0f}':<18}")
print(f"{'Trades/Year':<25} {'~7':<18} {f'~{int(total_trades/6)}':<18}")

print()
print('='*80)
print('Files saved:')
print('  - reports/relaxed_909_backtest.csv')
print('='*80)
