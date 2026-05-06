#!/usr/bin/env python3
"""Test downloading 1-minute data from Alpaca."""

import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))

# Check credentials from environment or .env file
api_key = os.getenv('ALPACA_API_KEY')
secret_key = os.getenv('ALPACA_SECRET')

# Try loading from .env file
if not api_key or not secret_key:
    env_path = Path('.env')
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    if key == 'ALPACA_API_KEY' and not api_key:
                        api_key = value
                    elif key == 'ALPACA_SECRET' and not secret_key:
                        secret_key = value

if not api_key or not secret_key:
    print("ERROR: ALPACA_API_KEY and ALPACA_SECRET not found")
    print("Options:")
    print("  1. Set environment variables:")
    print("     export ALPACA_API_KEY=your_key")
    print("     export ALPACA_SECRET=your_secret")
    print("  2. Or create .env file with:")
    print("     ALPACA_API_KEY=your_key")
    print("     ALPACA_SECRET=your_secret")
    sys.exit(1)

print(f"API Key: {api_key[:8]}...")

# Try to download test data
symbol = "AAPL"
date_str = "2024-03-15"  # Recent Friday

print(f"\nTesting 1-minute download for {symbol} on {date_str}...")

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    
    client = StockHistoricalDataClient(api_key, secret_key)
    
    date = datetime.strptime(date_str, "%Y-%m-%d").date()
    
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=datetime.combine(date, datetime.min.time()),
        end=datetime.combine(date, datetime.max.time()),
        feed="iex"  # Free tier
    )
    
    bars = client.get_stock_bars(request)
    
    if symbol in bars.data:
        bar_list = bars.data[symbol]
        print(f"✓ SUCCESS: Downloaded {len(bar_list)} 1-minute bars")
        
        # Show sample
        import polars as pl
        data = {
            'timestamp': [b.timestamp for b in bar_list[:5]],
            'open': [b.open for b in bar_list[:5]],
            'close': [b.close for b in bar_list[:5]],
        }
        df = pl.DataFrame(data)
        print("\nFirst 5 bars:")
        print(df)
        
        print(f"\nTime range: {bar_list[0].timestamp} to {bar_list[-1].timestamp}")
        
    else:
        print(f"✗ No data returned")
        print(f"Response: {bars}")
        
except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
