#!/usr/bin/env python3
"""Quick test of data provider fix."""
import sys
sys.path.insert(0, 'src')

from rl.data_provider_hybrid import get_data_provider, reset_data_provider
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Reset to get fresh provider
reset_data_provider()

print("=" * 60)
print("TESTING DATA PROVIDER FIX")
print("=" * 60)

provider = get_data_provider()
print(f"\nLoaded setups: {len(provider.csv_setups)} CSV + {len(provider.parquet_setups)} Parquet")

# Test 3 episodes
for i in range(3):
    print(f"\n{'='*60}")
    print(f"Episode {i+1}")
    print("=" * 60)
    
    success = provider.reset_episode()
    if success:
        print(f"✓ Symbol: {provider.current_symbol}")
        print(f"  Date: {provider.current_date}")
        print(f"  Start bar: {provider.start_bar_idx}")
        print(f"  Total bars: {provider.get_total_bars()}")
        
        obs = provider.get_observation()
        if obs is not None:
            print(f"  Observation shape: {obs.shape}")
        
        features = provider.get_state_features()
        print(f"  VWAP deviation: {features['vwap_deviation']:.1f}%")
        
        # Simulate a few steps
        for step in range(5):
            if not provider.step():
                break
        print(f"  Stepped 5 bars, now at: {provider.get_current_bar_index()}")
    else:
        print("✗ Failed to load episode")

print(f"\n{'='*60}")
print("TEST COMPLETE")
print("=" * 60)
