#!/usr/bin/env python3
import sys
from pathlib import Path

print(f"Cache dir exists: {Path('data/cache').exists()}")
print(f"Setups file exists: {Path('reports/relaxed_909_backtest.csv').exists()}")

from src.rl.data_provider import HistoricalDataProvider

dp = HistoricalDataProvider(
    intraday_data_dir='data/cache',
    setups_csv='reports/relaxed_909_backtest.csv'
)
print(f"Setup pairs: {len(dp.setup_pairs)}")
print(f"Available intraday symbols: {len(dp._symbol_file_map)}")

success = dp.reset_episode()
print(f"Reset success: {success}")

if success and dp.current_day:
    print(f"✓ SUCCESS: Loaded {len(dp.current_day)} bars")
    print(f"  Symbol: {dp.current_day.symbol}")
    print(f"  Date: {dp.current_day.date}")
    if len(dp.current_day.bars) > 0:
        print(f"  First bar: {dp.current_day.bars[0].timestamp}")
        print(f"  Last bar: {dp.current_day.bars[-1].timestamp}")
        print(f"  VWAP dev range: {dp.current_day.bars[0].vwap_deviation:.2f} to {dp.current_day.bars[-1].vwap_deviation:.2f}")
else:
    print("✗ FAILED")
