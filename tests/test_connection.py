#!/usr/bin/env python3
"""Quick connection test to Alpaca."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.alpaca_client import AlpacaClient

print("Testing Alpaca connection...")
print(f"API Key found: {'Yes' if os.getenv('ALPACA_API_KEY') else 'No'}")
print(f"Secret found: {'Yes' if os.getenv('ALPACA_SECRET') else 'No'}")

try:
    client = AlpacaClient()
    account = client.get_account()
    
    if account:
        print(f"\nConnected successfully!")
        print(f"Account ID: {account.get('id')}")
        print(f"Equity: ${account.get('equity', 0):,.2f}")
        print(f"Buying Power: ${account.get('buying_power', 0):,.2f}")
        print(f"Status: {account.get('status')}")
    else:
        print("\nFailed to fetch account info")
        
except Exception as e:
    print(f"\nConnection error: {e}")
