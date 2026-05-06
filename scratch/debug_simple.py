#!/usr/bin/env python3
import sys
print("STEP 1: Starting...", file=sys.stderr, flush=True)

from pathlib import Path
print("STEP 2: Pathlib imported", file=sys.stderr, flush=True)

cache_dir = Path("data/cache")
print(f"STEP 3: Cache dir exists: {cache_dir.exists()}", file=sys.stderr, flush=True)

files = list(cache_dir.glob("*_1min_*.parquet"))
print(f"STEP 4: Found {len(files)} files", file=sys.stderr, flush=True)

setups_file = Path("reports/relaxed_909_backtest.csv")
print(f"STEP 5: Setups file exists: {setups_file.exists()}", file=sys.stderr, flush=True)

print("STEP 6: Importing data_provider...", file=sys.stderr, flush=True)
from src.rl.data_provider import HistoricalDataProvider
print("STEP 7: Imported successfully", file=sys.stderr, flush=True)

print("STEP 8: Creating provider...", file=sys.stderr, flush=True)
dp = HistoricalDataProvider(
    intraday_data_dir='data/cache',
    setups_csv='reports/relaxed_909_backtest.csv'
)
print(f"STEP 9: Created. Setup pairs: {len(dp.setup_pairs)}", file=sys.stderr, flush=True)

print("STEP 10: Resetting episode...", file=sys.stderr, flush=True)
success = dp.reset_episode()
print(f"STEP 11: Reset success: {success}", file=sys.stderr, flush=True)

if success and dp.current_day:
    print(f"SUCCESS: {len(dp.current_day)} bars for {dp.current_day.symbol}")
else:
    print("FAILED to load day data")
