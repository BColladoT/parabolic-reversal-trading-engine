#!/usr/bin/env python3
"""Debug CSV loading."""
import sys
sys.path.insert(0, 'src')

import pandas as pd
from pathlib import Path
from rl.data_provider_hybrid import HybridDataProvider

# Load just first few rows
print("Loading CSV...")
df = pd.read_csv('reports/relaxed_909_backtest.csv')
print(f"Total rows: {len(df)}")

# Filter to profitable
profitable = df[df['pnl'] > 100]
print(f"Profitable rows: {len(profitable)}")

# Check first few
parquet_dir = Path('data/cache/1min_extended')

for i, row in profitable.head(5).iterrows():
    symbol = row['symbol']
    date_str = row['date']
    pnl = row['pnl']
    
    print(f"\n{i}: {symbol} on {date_str}, PnL: ${pnl:.2f}")
    
    # Check file exists
    data_file = parquet_dir / f"{symbol}.parquet"
    if not data_file.exists():
        matching_files = list(parquet_dir.glob(f"{symbol}_1min_*.parquet"))
        if matching_files:
            data_file = matching_files[0]
            print(f"  Found file: {data_file.name}")
        else:
            print(f"  No data file!")
            continue
    else:
        print(f"  Found file: {data_file.name}")
    
    # Now test VWAP validation
    import polars as pl
    from datetime import datetime
    
    try:
        df_pl = pl.read_parquet(data_file)
        date_val = datetime.strptime(date_str, '%Y-%m-%d').date()
        df_date = df_pl.filter(pl.col('timestamp').dt.date() == date_val)
        
        print(f"  Bars for date: {len(df_date)}")
        
        if len(df_date) == 0:
            print(f"  No data for this date!")
            continue
        
        # Check if VWAP column exists
        if 'vwap' not in df_date.columns:
            print(f"  No VWAP column!")
        else:
            print(f"  VWAP column exists")
        
        # Calculate VWAP manually
        provider = HybridDataProvider.__new__(HybridDataProvider)
        df_calc = provider._calculate_vwap(df_date)
        
        df_calc = df_calc.with_columns([
            ((pl.col('close') - pl.col('vwap')) / pl.col('vwap') * 100).alias('vwap_dev')
        ])
        
        max_dev = df_calc['vwap_dev'].abs().max()
        print(f"  Max VWAP deviation: {max_dev:.2f}%")
        print(f"  Passes 23% threshold: {max_dev >= 23.0}")
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
