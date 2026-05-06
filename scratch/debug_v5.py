#!/usr/bin/env python3
"""Debug V5 stop logic."""
import sys
sys.path.insert(0, '.')
from datetime import datetime
import pytz
from src.backtest.historical_tick_fetcher import tick_fetcher

et_tz = pytz.timezone('America/New_York')

symbol = "AMC"
date_str = "2021-06-02"

date = datetime.strptime(date_str, "%Y-%m-%d")
date = et_tz.localize(date)

tick_df = tick_fetcher.fetch_combined_tick_data(symbol, date, use_quotes=False)
bar_df = tick_fetcher.aggregate_trades_to_bars(tick_df, interval_seconds=60)

# Simulate V5 entry at 13:54
entry_time = None
entry_price = 40.87
stop_loss = entry_price * 1.03  # 3% stop

print(f"Entry: ${entry_price:.2f} at 13:54")
print(f"Stop: ${stop_loss:.2f}")
print()

for row in bar_df.to_dicts():
    ts = row['timestamp'].astimezone(et_tz)
    
    if ts.hour == 13 and ts.minute >= 54 or ts.hour >= 14:
        high = row['high']
        close = row['close']
        
        # Check if stop would trigger
        if high >= stop_loss:
            print(f"{ts.strftime('%H:%M')}: High=${high:.2f} >= Stop=${stop_loss:.2f} -> STOP HIT!")
            break
        else:
            print(f"{ts.strftime('%H:%M')}: High=${high:.2f} < Stop=${stop_loss:.2f}")
