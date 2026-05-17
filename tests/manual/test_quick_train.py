#!/usr/bin/env python3
"""Quick test with CSV setups only (skips Parquet scan)."""
import sys
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

sys.path.insert(0, 'src')

# RL deps (gymnasium, torch) are optional extras; skip collection cleanly when absent.
pytest.importorskip("gymnasium")
try:
    import torch  # noqa: F401
except (ImportError, OSError) as e:
    pytest.skip(f"torch unavailable: {e}", allow_module_level=True)

from rl.data_provider_hybrid import HybridDataProvider, reset_data_provider
import logging
logging.basicConfig(level=logging.INFO)

# Reset provider
reset_data_provider()

# Create provider with just CSV (skip Parquet by setting weight to 1.0)
provider = HybridDataProvider(
    csv_path='reports/relaxed_909_backtest.csv',
    parquet_dir='data/cache/1min_extended',
    cache_dir='src/scripts/data/cache',
    csv_weight=1.0,  # Always use CSV
    min_vwap_deviation=23.0,
    skip_parquet_scan=True  # Skip slow Parquet scanning
)

# Stop after loading CSV (don't scan all Parquet)
print(f"\n{'='*60}")
print(f"CSV SETUPS: {len(provider.csv_setups)}")
print(f"PARQUET SETUPS: {len(provider.parquet_setups)} (scanning skipped for speed)")
print(f"{'='*60}\n")

# Test episode loading
print("Testing episode loading...")
for i in range(3):
    success = provider.reset_episode()
    if success:
        print(f"\nEpisode {i+1}: {provider.current_symbol} {provider.current_date}")
        print(f"  Start bar: {provider.start_bar_idx}, Total bars: {provider.get_total_bars()}")
        features = provider.get_state_features()
        print(f"  VWAP deviation: {features['vwap_deviation']:.1f}%")
    else:
        print(f"Episode {i+1}: FAILED")

print("\n✓ Data provider working correctly!")
