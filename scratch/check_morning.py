#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from datetime import datetime
import pytz
from src.backtest.historical_tick_fetcher import tick_fetcher

et_tz = pytz.timezone('America/New_York')

for symbol, date_str in [("AMC", "2021-06-02")]:
    date = datetime.strptime(date_str, "%Y-%m-%d")
    date = et_tz.localize(date)
    
    tick_df = tick_fetcher.fetch_combined_tick_data(symbol, date, use_quotes=False)
    bar_df = tick_fetcher.aggregate_trades_to_bars(tick_df, interval_seconds=60)
    
    print(f"{symbol} {date_str} - First 20 bars:")
    for i, row in enumerate(bar_df.to_dicts()[:20]):
        ts = row['timestamp'].astimezone(et_tz)
        print(f"{ts.strftime('%H:%M')}: Open={row['open']:.2f}, High={row['high']:.2f}, Low={row['low']:.2f}, Close={row['close']:.2f}, Vol={row['volume']}")
