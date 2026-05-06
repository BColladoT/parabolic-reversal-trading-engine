#!/usr/bin/env python3
import pickle
from pathlib import Path

# Load the setups that were found
cache_path = Path('data/cache/setups/setups_20230101_20241231.pkl')
if cache_path.exists():
    with open(cache_path, 'rb') as f:
        setups = pickle.load(f)
    
    print(f'Found {len(setups)} parabolic setups (2023-2024):')
    print(f"{'Date':<12} {'Symbol':<8} {'Gain':<8} {'Close':<10} {'Volume':<12} {'Days Up'}")
    print('-' * 65)
    for s in setups:
        date_str = s.date.strftime('%Y-%m-%d')
        print(f'{date_str:<12} {s.symbol:<8} {s.gain_percent:>6.1f}%  ${s.day_close:<9.2f} {s.day_volume:>11,}  {s.days_up}')
    
    print(f"\nNote: These were filtered out because they didn't meet 'First Red Day' criteria:")
    print("  - Need 2-5 consecutive green days")
    print("  - Need 30%+ gain over prior 5 days")
    print("\nTo test these setups anyway, run:")
    print("  python run_historical_backtest.py --symbol <SYMBOL> --date <DATE>")
else:
    print('No cache file found')
