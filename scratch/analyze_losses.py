"""
Analyze losing trades from backtest to identify patterns for ML risk management.
"""
import pandas as pd
import numpy as np

# Load backtest results
df = pd.read_csv('reports/full_3527_backtest_results.csv')

print("="*80)
print("LOSING TRADES ANALYSIS FOR ML RISK MANAGEMENT")
print("="*80)

# Filter losing trades (trades taken but resulted in loss)
trades_df = df[df['trades'] > 0].copy()
losing_trades = trades_df[trades_df['pnl'] < 0].copy()
winning_trades = trades_df[trades_df['pnl'] > 0].copy()

print(f"\nTotal setups scanned: {len(df)}")
print(f"Setups with trades taken: {len(trades_df)}")
print(f"Losing trades: {len(losing_trades)} ({len(losing_trades)/len(trades_df)*100:.1f}% of trades)")
print(f"Winning trades: {len(winning_trades)} ({len(winning_trades)/len(trades_df)*100:.1f}% of trades)")

print("\n" + "="*80)
print("TOP 20 WORST LOSING TRADES")
print("="*80)
worst_losses = losing_trades.nsmallest(20, 'pnl')[['symbol', 'date', 'pnl', 'gain_pct', 'days_up', 'volume']]
print(worst_losses.to_string(index=False))

print("\n" + "="*80)
print("LOSING TRADE STATISTICS")
print("="*80)
print(f"Average loss per trade: ${losing_trades['pnl'].mean():,.2f}")
print(f"Median loss per trade: ${losing_trades['pnl'].median():,.2f}")
print(f"Worst single loss: ${losing_trades['pnl'].min():,.2f}")
print(f"Total losses: ${losing_trades['pnl'].sum():,.2f}")

print("\n" + "="*80)
print("CHARACTERISTICS OF LOSING TRADES vs WINNING TRADES")
print("="*80)
print(f"\nAverage day gain % (losing trades): {losing_trades['gain_pct'].mean():.1f}%")
print(f"Average day gain % (winning trades): {winning_trades['gain_pct'].mean():.1f}%")

print(f"\nAverage days up (losing trades): {losing_trades['days_up'].mean():.2f}")
print(f"Average days up (winning trades): {winning_trades['days_up'].mean():.2f}")

print(f"\nAverage volume (losing trades): {losing_trades['volume'].mean():,.0f}")
print(f"Average volume (winning trades): {winning_trades['volume'].mean():,.0f}")

# Distribution by loss size
print("\n" + "="*80)
print("LOSS DISTRIBUTION")
print("="*80)
bins = [-float('inf'), -10000, -5000, -2000, -1000, 0]
labels = ['>$10K', '$5K-$10K', '$2K-$5K', '$1K-$2K', '<$1K']
losing_trades['loss_bucket'] = pd.cut(losing_trades['pnl'], bins=bins, labels=labels)
print(losing_trades['loss_bucket'].value_counts().sort_index())

# Save losing trades for further analysis
losing_trades[['symbol', 'date', 'pnl', 'gain_pct', 'days_up', 'volume']].to_csv(
    'reports/losing_trades_analysis.csv', index=False
)
print("\n[SAVED] Losing trades saved to reports/losing_trades_analysis.csv")
