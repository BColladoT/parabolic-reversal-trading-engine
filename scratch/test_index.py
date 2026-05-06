#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')
import logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

from rl.data_provider_hybrid import HybridDataProvider

provider = HybridDataProvider(
    csv_path='reports/relaxed_909_backtest.csv',
    parquet_dir='data/cache/1min_extended',
    cache_dir='src/scripts/data/cache'
)
print(f'CSV setups: {len(provider.csv_setups)}')
print(f'Parquet setups: {len(provider.parquet_setups)}')
