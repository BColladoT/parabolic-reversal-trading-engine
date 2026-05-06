#!/usr/bin/env python3
"""
Explain why setups are being filtered out.
"""
import pickle
from pathlib import Path
from datetime import datetime

# Load the setups
print("="*80)
print("UNDERSTANDING THE FILTERING ISSUE")
print("="*80)

cache_file = Path("data/cache/setups/setups_20190101_20241231.pkl")
if not cache_file.exists():
    print("No cache found. Run a scan first.")
    exit()

with open(cache_file, 'rb') as f:
    all_setups = pickle.load(f)

print(f"\nTotal setups found: {len(all_setups)}")
print("\nAll setups (before filtering):")
print(f"{'Date':<12} {'Symbol':<8} {'Gain':<10} {'Days Up':<10} {'Prior 5D':<10} {'Volume':<15}")
print("-" * 80)

for s in all_setups:
    date_str = s.date.strftime('%Y-%m-%d')
    print(f"{date_str:<12} {s.symbol:<8} {s.gain_percent:>8.1f}%  "
          f"{s.days_up:<10} {s.prior_gain_percent:>8.1f}%  {s.day_volume:>14,}")

print(f"\n{'='*80}")
print("FILTERING CRITERIA (Quality First Red Day):")
print("="*80)
print("1. Days Up: 2-5 consecutive (many have only 1)")
print("2. Prior 5D Gain: 30%+ (many have less)")
print("3. Price: $1-50")
print("4. Volume: >100k")

# Apply filtering manually to show why
filtered = []
for s in all_setups:
    reasons = []
    
    if not (2 <= s.days_up <= 5):
        reasons.append(f"days_up={s.days_up} (need 2-5)")
    
    if s.prior_gain_percent < 30:
        reasons.append(f"prior_gain={s.prior_gain_percent:.1f}% (need 30%+)")
    
    if not (1.0 <= s.day_close <= 50.0):
        reasons.append(f"price=${s.day_close:.2f} (need $1-50)")
    
    if s.day_volume < 100000:
        reasons.append(f"volume={s.day_volume:,} (need 100k+)")
    
    if reasons:
        status = "FILTERED: " + ", ".join(reasons)
    else:
        status = "QUALITY SETUP ✓"
        filtered.append(s)
    
    print(f"\n{s.symbol} {s.date.strftime('%Y-%m-%d')}: {status}")

print(f"\n{'='*80}")
print(f"AFTER FILTERING: {len(filtered)} quality setups")
print(f"{'='*80}")

if len(filtered) < len(all_setups):
    print("\nTo test ALL setups without filtering, run:")
    print("  python run_all_setups.py")
    print("\nOr adjust criteria in config/settings.yaml")
