"""
Scan with RELAXED criteria to find MORE parabolic setups,
but keep V5's STRICT entry criteria for trading.
"""
import sys
from pathlib import Path
from datetime import datetime
import pickle

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.backtest.historical_screener import HistoricalParabolicScreener, ParabolicSetup
from src.backtest.extended_universe import ALL_MICRO_CAP_SYMBOLS
from src.utils.logger import logger


def scan_with_relaxed_criteria():
    """Scan with lower thresholds to find more setups."""
    
    print("="*70)
    print("SCANNING WITH RELAXED CRITERIA")
    print("="*70)
    print("\nRelaxed Discovery Criteria:")
    print("  - Min gain: 30% (was 50%)")
    print("  - Min volume: 2x average (was 3x)")
    print("  - Price range: $0.20 - $100 (was $0.50-$50)")
    print("  - Keep V5's STRICT entry (2-of-3 criteria)")
    print()
    
    screener = HistoricalParabolicScreener()
    
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    # RELAXED scan criteria
    setups = screener.scan_for_parabolic_setups(
        symbols=ALL_MICRO_CAP_SYMBOLS,
        start_date=start_date,
        end_date=end_date,
        min_gain_percent=30.0,      # Was 50% - LOWER
        max_gain_percent=500.0,
        min_volume_multiplier=2.0,  # Was 3.0 - LOWER
        use_cache=False  # Force fresh scan
    )
    
    print(f"\n[SCAN RESULTS]")
    print(f"Total setups found: {len(setups)}")
    
    # Quality filter (relaxed)
    quality_setups = screener.filter_quality_setups(
        setups,
        min_days_up=1,  # Was 2 - allow single-day parabolics
        max_days_up=5,
        min_prior_gain=20.0,  # Was 30% - LOWER
        min_volume=50000,     # Was 100K - LOWER
        min_gain_percent=30.0 # Was 50% - LOWER
    )
    
    print(f"Quality setups: {len(quality_setups)}")
    
    # Analyze distribution
    analysis = screener.analyze_setup_distribution(quality_setups)
    print(f"\n[DISTRIBUTION]")
    print(f"Avg gain: {analysis['avg_gain_percent']:.1f}%")
    print(f"Median gain: {analysis['median_gain_percent']:.1f}%")
    print(f"Avg days up: {analysis['avg_days_up']:.1f}")
    print(f"Gain range: {analysis['gain_range'][0]:.1f}% - {analysis['gain_range'][1]:.1f}%")
    
    # Top symbols
    print(f"\n[TOP SYMBOLS BY FREQUENCY]")
    for sym, count in analysis['top_symbols'][:10]:
        print(f"  {sym}: {count} setups")
    
    # Compare to previous
    old_cache = Path("data/cache/setups/setups_20190101_20241231.pkl")
    if old_cache.exists():
        with open(old_cache, 'rb') as f:
            old_setups = pickle.load(f)
        print(f"\n[COMPARISON]")
        print(f"Previous scan (50% gain): {len(old_setups)} setups")
        print(f"New scan (30% gain): {len(quality_setups)} setups")
        print(f"Increase: {len(quality_setups) - len(old_setups):+d} ({(len(quality_setups)/len(old_setups)-1)*100:+.1f}%)")
        print(f"\nExpected trades with V5 (50% conversion): ~{int(len(quality_setups) * 0.5)}")
    
    # Export
    output = screener.export_setups_for_backtest(quality_setups, "reports/relaxed_setups.csv")
    print(f"\nExported to: {output}")
    
    # Cache for backtest
    cache_path = Path("data/cache/setups/setups_relaxed_30pct.pkl")
    with open(cache_path, 'wb') as f:
        pickle.dump(quality_setups, f)
    print(f"Cached to: {cache_path}")
    
    print("\n" + "="*70)
    print("RECOMMENDATION:")
    print("  Run backtest with these relaxed setups but keep V5 entry criteria")
    print("  This should significantly increase trade count while maintaining win rate")
    print("="*70)


if __name__ == "__main__":
    scan_with_relaxed_criteria()
