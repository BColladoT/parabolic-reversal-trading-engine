#!/bin/bash
cd /mnt/c/quant_trading
source venv_wsl/bin/activate
python3 -c "
import polars as pl
df = pl.read_parquet('data/cache/AACG_1min_20190101_20241231.parquet')
print('Columns:', df.columns)
print('\\nSample:')
print(df.head(3))
"
