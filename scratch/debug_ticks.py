#!/usr/bin/env python3
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

print("Raw ticks around 13:54:")
for row in tick_df.filter(tick_df['price'] > 0).to_dicts():
    ts = row['timestamp'].astimezone(et_tz)
    if ts.hour == 13 and ts.minute >= 50:
        print(f"{ts.strftime('%H:%M:%S')}: ${row['price']:.2f}, Qty={row['quantity']}")
