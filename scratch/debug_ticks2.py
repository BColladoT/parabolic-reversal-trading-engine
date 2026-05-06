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

print("Columns:", tick_df.columns)
print("\nFirst 10 rows:")
print(tick_df.head(10))
