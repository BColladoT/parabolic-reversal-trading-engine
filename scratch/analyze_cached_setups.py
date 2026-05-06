"""Analyze cached relaxed setups"""
import pickle
import pandas as pd
from pathlib import Path

cache_path = Path('data/cache/setups/setups_relaxed_full_2019_2024.pkl')
if cache_path.exists():
    with open(cache_path, 'rb') as f:
        setups = pickle.load(f)
    
    print(f'Total setups in cache: {len(setups)}')
    
    # Create dataframe for analysis
    data = []
    for s in setups:
        data.append({
            'symbol': s.symbol,
            'date': s.date.strftime('%Y-%m-%d'),
            'gain_pct': s.gain_percent,
            'days_up': s.days_up,
            'volume': s.day_volume,
            'open': s.day_open,
            'high': s.day_high,
            'close': s.day_close
        })
    
    df = pd.DataFrame(data)
    
    print('\n[SETUP STATISTICS]')
    print(f"Gain range: {df['gain_pct'].min():.1f}% - {df['gain_pct'].max():.1f}%")
    print(f"Avg gain: {df['gain_pct'].mean():.1f}%")
    print(f"Median gain: {df['gain_pct'].median():.1f}%")
    print(f"Days up range: {df['days_up'].min()} - {df['days_up'].max()}")
    print(f"Avg days up: {df['days_up'].mean():.1f}")
    
    print('\n[TOP SYMBOLS BY FREQUENCY]')
    top_symbols = df['symbol'].value_counts().head(10)
    for sym, count in top_symbols.items():
        print(f'  {sym}: {count} setups')
    
    print('\n[GAIN DISTRIBUTION]')
    print(f"  30-40%: {len(df[(df.gain_pct >= 30) & (df.gain_pct < 40)])} setups")
    print(f"  40-50%: {len(df[(df.gain_pct >= 40) & (df.gain_pct < 50)])} setups")
    print(f"  50-75%: {len(df[(df.gain_pct >= 50) & (df.gain_pct < 75)])} setups")
    print(f"  75-100%: {len(df[(df.gain_pct >= 75) & (df.gain_pct < 100)])} setups")
    print(f"  100%+: {len(df[df.gain_pct >= 100])} setups")
    
    print('\n[YEAR DISTRIBUTION]')
    df['year'] = pd.to_datetime(df['date']).dt.year
    for year in sorted(df['year'].unique()):
        count = len(df[df['year'] == year])
        print(f'  {year}: {count} setups')
    
    # Compare to original
    original_cache = Path('data/cache/setups/setups_20190101_20241231.pkl')
    if original_cache.exists():
        with open(original_cache, 'rb') as f:
            original_setups = pickle.load(f)
        print(f"\n[COMPARISON]")
        print(f"Original (50% gain): {len(original_setups)} setups")
        print(f"Relaxed (30% gain): {len(setups)} setups")
        print(f"Increase: {len(setups) - len(original_setups)} ({(len(setups)/len(original_setups)-1)*100:.0f}% more)")
        
        # Estimate trades
        print(f"\n[TRADE PROJECTION]")
        print(f"Original: ~40 trades (50% of 79 quality)")
        print(f"Relaxed: ~{int(len(setups) * 0.5)} trades (est. 50% conversion)")
        
else:
    print('Cache file not found')
