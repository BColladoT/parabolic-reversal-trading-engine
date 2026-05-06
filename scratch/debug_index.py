#!/usr/bin/env python3
"""Debug index building."""
import sys
sys.path.insert(0, 'src')

from rl.data_provider_hybrid import HybridDataProvider
from pathlib import Path
import pandas as pd

# Create provider with explicit paths
provider = HybridDataProvider(
    csv_path="reports/relaxed_909_backtest.csv",
    parquet_dir="data/cache/1min_extended",
    cache_dir="src/scripts/data/cache"
)

print(f"CSV path: {provider.csv_path}")
print(f"Parquet dir: {provider.parquet_dir}")
print(f"Exists: {provider.parquet_dir.exists()}")

# Check first CSV row
df = pd.read_csv(provider.csv_path)
profitable = df[df['pnl'] > 100]
row = profitable.iloc[0]
print(f"\nFirst profitable row:")
print(f"  Symbol: {row['symbol']}")
print(f"  Date: {row['date']}")
print(f"  PnL: {row['pnl']}")

# Test validation directly
symbol = row['symbol']
date_str = row['date']

print(f"\nTesting VWAP validation for {symbol} {date_str}...")
result = provider._validate_vwap_in_data(symbol, date_str)
print(f"Result: {result}")
