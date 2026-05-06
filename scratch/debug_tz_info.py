#!/usr/bin/env python3
"""Check timezone info in cache."""
import sys
sys.path.insert(0, '.')
from datetime import datetime
import pytz
import polars as pl

et_tz = pytz.timezone('America/New_York')

# Load the parquet file directly
cache_path = "data/cache/ticks/AMC_trades_20210602.parquet"
df = pl.read_parquet(cache_path)

print("First 5 timestamps:")
for row in df.head(5).to_dicts():
    ts = row['timestamp']
    print(f"  Raw: {ts}, tzinfo: {ts.tzinfo}")
    if ts.tzinfo:
        ts_et = ts.astimezone(et_tz)
        print(f"  ET:  {ts_et}")
    else:
        print(f"  No timezone info!")
