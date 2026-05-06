#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from datetime import datetime
import pytz
from src.backtest.historical_tick_fetcher import tick_fetcher

et_tz = pytz.timezone('America/New_York')
test_cases = [
    ("WWR", "2020-10-05"),
    ("AMC", "2021-06-02"),
]

for symbol, date_str in test_cases:
    date = datetime.strptime(date_str, "%Y-%m-%d")
    date = et_tz.localize(date)
    
    tick_df = tick_fetcher.fetch_combined_tick_data(symbol, date, use_quotes=False)
    bar_df = tick_fetcher.aggregate_trades_to_bars(tick_df, interval_seconds=60)
    
    print(f"\n{symbol} {date_str}")
    open_price = bar_df["open"][0]
    high_price = bar_df["high"].max()
    print(f"Open: ${open_price:.2f}, High: ${high_price:.2f}")
    
    for row in bar_df.to_dicts():
        ts = row['timestamp'].astimezone(et_tz)
        if ts.hour == 13 or (ts.hour == 14 and ts.minute < 30):
            close = row['close']
            vol = row['volume']
            high = row['high']
            low = row['low']
            print(f"  {ts.strftime('%H:%M')}: Close=${close:.2f}, Vol={vol}, Range=${low:.2f}-${high:.2f}")
