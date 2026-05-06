#!/usr/bin/env python3
import sys
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

print("Starting data provider debug...", flush=True)

from src.rl.data_provider import HistoricalDataProvider
from pathlib import Path

print("Imports successful", flush=True)

# Check if files exist
cache_dir = Path("data/cache")
print(f"Cache dir exists: {cache_dir.exists()}", flush=True)

# List a few files
files = list(cache_dir.glob("*_1min_*.parquet"))[:5]
print(f"Found {len(files)} sample files:", flush=True)
for f in files:
    print(f"  {f.name}", flush=True)

# Check setups file
setups_file = Path("reports/relaxed_909_backtest.csv")
print(f"Setups file exists: {setups_file.exists()}", flush=True)

print("\nInitializing data provider...", flush=True)
try:
    dp = HistoricalDataProvider(
        intraday_data_dir='data/cache',
        setups_csv='reports/relaxed_909_backtest.csv'
    )
    print(f"Data provider initialized", flush=True)
    print(f"  Setup pairs: {len(dp.setup_pairs)}", flush=True)
    print(f"  Available symbols: {len(dp._available_intraday)}", flush=True)
    
    print("\nAttempting reset_episode()...", flush=True)
    success = dp.reset_episode()
    print(f"Reset success: {success}", flush=True)
    
    if success and dp.current_day:
        print(f"✓ SUCCESS: Loaded {len(dp.current_day)} bars", flush=True)
        print(f"  Symbol: {dp.current_day.symbol}", flush=True)
        print(f"  First bar: {dp.current_day.bars[0].timestamp}", flush=True)
        print(f"  Last bar: {dp.current_day.bars[-1].timestamp}", flush=True)
    else:
        print("✗ FAILED to load day data", flush=True)
        
except Exception as e:
    print(f"ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
