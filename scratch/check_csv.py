#!/usr/bin/env python3
import pandas as pd

df = pd.read_csv('reports/relaxed_909_backtest.csv')
print('Columns:', df.columns.tolist())
print('Total rows:', len(df))
print('Rows with pnl > 100:', len(df[df['pnl'] > 100]))
print('Max PnL:', df['pnl'].max())
print('Min PnL:', df['pnl'].min())
print()
print('Sample profitable row:')
profitable = df[df['pnl'] > 100]
if len(profitable) > 0:
    print(profitable.iloc[0])
else:
    print("No profitable rows found!")
