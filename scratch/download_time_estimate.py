#!/usr/bin/env python3
"""Calculate download time for extended universe."""

total_symbols = 3527
years = 6
trading_days_per_year = 252
bars_per_day = 390

# Alpaca limits
bars_per_request = 10000  # Max bars per API call
rate_limit = 200  # requests per minute
delay_between_symbols = 0.5  # seconds

# Calculate chunks per symbol
bars_per_symbol = years * trading_days_per_year * bars_per_day
chunks_per_symbol = bars_per_symbol / bars_per_request

print(f"=== Download Estimate for Extended Universe ===")
print(f"\nParameters:")
print(f"  Total symbols: {total_symbols:,}")
print(f"  Years per symbol: {years}")
print(f"  Trading days per year: ~{trading_days_per_year}")
print(f"  Bars per day: ~{bars_per_day}")
print(f"\nCalculations:")
print(f"  Total bars per symbol: {bars_per_symbol:,}")
print(f"  API calls (chunks) per symbol: {chunks_per_symbol:.1f}")
print(f"  Total API calls: {total_symbols * chunks_per_symbol:,.0f}")

# Time calculation
time_per_chunk = 60 / rate_limit  # seconds per API call
total_api_time = (total_symbols * chunks_per_symbol * time_per_chunk) / 3600
total_delay_time = (total_symbols * delay_between_symbols) / 3600
total_hours = total_api_time + total_delay_time
total_days = total_hours / 24

print(f"\nTime Estimate:")
print(f"  Time per API call: {time_per_chunk:.2f}s")
print(f"  API call time: {total_api_time:.1f} hours")
print(f"  Delay time: {total_delay_time:.1f} hours")
print(f"  TOTAL TIME: {total_hours:.1f} hours ({total_days:.1f} days)")
print(f"\nWith 50% retry rate: {total_hours * 1.5:.1f} hours ({total_days * 1.5:.1f} days)")

print(f"\n=== Alternative: Day-by-Day Download (OLD METHOD) ===")
day_by_day_calls = total_symbols * years * trading_days_per_year
day_by_day_hours = (day_by_day_calls * time_per_chunk) / 3600
print(f"  API calls: {day_by_day_calls:,}")
print(f"  Time: {day_by_day_hours:,.0f} hours ({day_by_day_hours/24:.0f} days)")
print(f"  This is why we use multi-year chunks!")
