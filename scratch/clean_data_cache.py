"""
Clean the data provider cache to force re-scan of Parquet files.
Use this if:
1. You've added new Parquet files
2. You want to change date_range or min_bars_per_day
3. The cache seems corrupted
"""

import shutil
from pathlib import Path

cache_file = Path("data/cache/trading_days_index.pkl")

if cache_file.exists():
    print(f"Removing cache file: {cache_file}")
    cache_file.unlink()
    print("✅ Cache cleared. Next run will rebuild the index.")
else:
    print("No cache file found. Index will be built on next run.")
