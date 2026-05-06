"""
Test script to verify the new data provider loads all Parquet files correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rl.data_provider import get_data_provider, reset_data_provider
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_data_provider():
    """Test that data provider loads all trading days."""
    
    print("=" * 70)
    print("TESTING DATA PROVIDER - ALL PARQUET FILES")
    print("=" * 70)
    
    # Reset any existing provider
    reset_data_provider()
    
    # Initialize data provider
    print("\n1. Initializing data provider...")
    provider = get_data_provider(
        intraday_data_dir="data/cache/1min_extended",
        cache_index_path="data/cache/trading_days_index.pkl",
        date_range=("2020-01-01", "2024-12-31"),
        min_bars_per_day=100
    )
    
    # Check statistics
    print(f"\n2. Data Provider Statistics:")
    print(f"   - Files scanned: {provider.stats['files_scanned']}")
    print(f"   - Symbols with data: {provider.stats['symbols_found']}")
    print(f"   - Total trading days: {provider.stats['trading_days_found']}")
    print(f"   - Unique dates: {len(provider.stats['dates_loaded'])}")
    
    if len(provider.setup_pairs) == 0:
        print("\n❌ ERROR: No trading days loaded!")
        return False
        
    # Test loading a few random episodes
    print(f"\n3. Testing episode loading (loading 3 random trading days):")
    for i in range(3):
        success = provider.reset_episode()
        if success and provider.current_day:
            day = provider.current_day
            print(f"   Episode {i+1}: {day.symbol} on {day.date.date()} "
                  f"- {len(day)} bars")
        else:
            print(f"   Episode {i+1}: ❌ Failed to load")
            
    # Show date range
    if provider.stats['dates_loaded']:
        sorted_dates = sorted(provider.stats['dates_loaded'])
        print(f"\n4. Date Range:")
        print(f"   - Earliest: {sorted_dates[0]}")
        print(f"   - Latest: {sorted_dates[-1]}")
        
    # Show sample of symbols
    symbols = list(set(p[0] for p in provider.setup_pairs))
    print(f"\n5. Sample Symbols ({min(10, len(symbols))} of {len(symbols)}):")
    for sym in sorted(symbols)[:10]:
        count = sum(1 for p in provider.setup_pairs if p[0] == sym)
        print(f"   - {sym}: {count} trading days")
        
    print(f"\n6. Setup Pairs Sample (first 10):")
    for sym, date in provider.setup_pairs[:10]:
        print(f"   - {sym} on {date}")
        
    print("\n" + "=" * 70)
    print(f"✅ Data provider test complete!")
    print(f"   Total trading days available: {len(provider.setup_pairs)}")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    success = test_data_provider()
    sys.exit(0 if success else 1)
