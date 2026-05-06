#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from datetime import datetime
import pytz
from src.backtest.historical_tick_fetcher import tick_fetcher

et_tz = pytz.timezone('America/New_York')

# Debug AMC 2021-06-02 - what happens after entry at 13:54 @ $40.87
symbol = "AMC"
date_str = "2021-06-02"

date = datetime.strptime(date_str, "%Y-%m-%d")
date = et_tz.localize(date)

tick_df = tick_fetcher.fetch_combined_tick_data(symbol, date, use_quotes=False)
bar_df = tick_fetcher.aggregate_trades_to_bars(tick_df, interval_seconds=60)

print(f"{symbol} {date_str} - After 13:54 entry at $40.87 (stop = $41.69):")
print()

# Entry: 13:54 @ $40.87, stop = $41.69 (2% above)
stop_price = 40.87 * 1.02

for row in bar_df.to_dicts():
    ts = row['timestamp'].astimezone(et_tz)
    if ts.hour == 13 and ts.minute >= 54 or ts.hour >= 14:
        high = row['high']
        low = row['low']  
        close = row['close']
        vol = row['volume']
        triggered = "STOP!" if high >= stop_price else ""
        print(f"{ts.strftime('%H:%M')}: Low=${low:.2f}, High=${high:.2f}, Close=${close:.2f}, Vol={vol:.0f} {triggered}")
