#!/usr/bin/env python3
import pickle
with open('src/scripts/data/cache/hybrid_index.pkl', 'rb') as f:
    idx = pickle.load(f)
print(f'CSV setups: {len(idx["csv_setups"])}')
print(f'Parquet setups: {len(idx["parquet_setups"])}')
print(f'Built at: {idx.get("built_at", "unknown")}')
