#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

from rl.data_provider_hybrid import HybridDataProvider

# Create provider
provider = HybridDataProvider(
    csv_path='reports/relaxed_909_backtest.csv',
    parquet_dir='data/cache/1min_extended',
    cache_dir='src/scripts/data/cache',
    min_vwap_deviation=23.0
)

# Test validation on first profitable row
import pandas as pd
df = pd.read_csv('reports/relaxed_909_backtest.csv')
profitable = df[df['pnl'] > 100]

print("Testing first 5 profitable setups:")
for i, row in profitable.head(5).iterrows():
    symbol = row['symbol']
    date_str = row['date']
    pnl = row['pnl']
    
    result = provider._validate_vwap_in_data(symbol, date_str)
    print(f"  {symbol} {date_str}: PnL=${pnl:.0f} -> {'PASS' if result else 'FAIL'}")
