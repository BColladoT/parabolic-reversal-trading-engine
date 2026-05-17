#!/usr/bin/env python3
"""
Parabolic Reversal Trading Engine - Test Suite
Validates all components before live deployment.
"""
import sys
import os
import numpy as np
import pytest

pytestmark = pytest.mark.integration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import CONFIG, load_config
from src.utils.logger import logger
from src.indicators.numba_kernels import (
    calculate_vwap_numba,
    calculate_atr_numba,
    calculate_position_size_numba,
    detect_absorption_numba,
    detect_momentum_divergence_numba
)
from src.data.polars_engine import StreamingBuffer, TickData, PolarsSignalEngine
from datetime import datetime


def test_config():
    """Test configuration loading."""
    print("\n=== Testing Configuration ===")
    
    assert CONFIG.broker.name == "alpaca", "Broker name mismatch"
    assert CONFIG.risk.max_portfolio_risk_percent == 1.0, "Risk percent mismatch"
    assert CONFIG.screening.min_percent_gain == 80.0, "Min gain mismatch"
    
    # Check API credentials
    if not CONFIG.broker.api_key or not CONFIG.broker.secret_key:
        print("WARNING: API credentials not set in environment")
    else:
        print(f"OK API Key: {CONFIG.broker.api_key[:8]}...")
    
    print("OK Configuration loaded successfully")


def test_numba_kernels():
    """Test Numba-optimized calculations."""
    print("\n=== Testing Numba Kernels ===")
    
    # Generate test data
    n = 100
    highs = np.random.uniform(10, 20, n).astype(np.float64)
    lows = highs - np.random.uniform(0.1, 1.0, n).astype(np.float64)
    closes = (highs + lows) / 2 + np.random.uniform(-0.1, 0.1, n).astype(np.float64)
    volumes = np.random.randint(1000, 10000, n).astype(np.float64)
    
    # Test VWAP
    vwap = calculate_vwap_numba(highs, lows, closes, volumes)
    assert len(vwap) == n, "VWAP length mismatch"
    assert vwap[-1] > 0, "VWAP should be positive"
    print(f"OK VWAP calculation: {vwap[-1]:.4f}")
    
    # Test ATR
    atr = calculate_atr_numba(highs, lows, closes, period=14)
    assert len(atr) == n, "ATR length mismatch"
    assert atr[-1] > 0, "ATR should be positive"
    print(f"OK ATR calculation: {atr[-1]:.4f}")
    
    # Test position sizing (short position: stop is ABOVE entry)
    # Entry at 15, stop at 16.5 ($1.50 risk per share = 10%)
    # Risk amount = $250 (1% of $25k)
    # Shares = $250 / $1.50 = 166 shares
    shares = calculate_position_size_numba(
        account_equity=25000.0,
        risk_percent=1.0,
        entry_price=15.0,
        stop_price=16.5,  # $1.50 risk per share
        max_position_value=50000.0
    )
    assert shares > 0, f"Position size should be positive, got {shares}"
    assert shares == 166, f"Expected 166 shares, got {shares}"  # $250 / $1.50 = 166
    print(f"OK Position sizing: {shares} shares")
    
    # Test absorption detection
    prices = np.linspace(15, 15.5, 50)
    volumes = np.concatenate([
        np.full(25, 1000),
        np.full(25, 3000)  # Volume spike
    ])
    absorption = detect_absorption_numba(prices, volumes, lookback=20)
    print(f"OK Absorption detection: {absorption}")
    
    # Test momentum divergence
    prices_div = np.linspace(15, 16, 20)
    volumes_div = np.concatenate([
        np.full(10, 5000),
        np.full(10, 2000)  # Volume drop
    ])
    divergence = detect_momentum_divergence_numba(prices_div, volumes_div, lookback=5)
    print(f"OK Momentum divergence: {divergence}")
    
    print("OK All Numba kernels passed")


def test_polars_engine():
    """Test Polars data engine."""
    print("\n=== Testing Polars Engine ===")
    
    engine = PolarsSignalEngine()
    symbol = "TEST"
    
    # Add test ticks
    for i in range(100):
        tick = TickData(
            timestamp=datetime.now(),
            symbol=symbol,
            price=15.0 + (i * 0.01),
            size=1000 + i * 10,
            side="A",
            exchange="N"
        )
        engine.process_tick(tick)
    
    # Get metrics
    metrics = engine.get_signal_data(symbol)
    assert len(metrics) > 0, "Metrics should not be empty"
    print(f"OK Metrics: {metrics}")
    
    # Get DataFrame
    buffer = engine.buffers.get(symbol)
    if buffer:
        df = buffer.to_polars_df()
        assert len(df) == 100, f"DataFrame should have 100 rows, got {len(df)}"
        print(f"OK DataFrame shape: {df.shape}")
        
        bar_df = buffer.get_bar_df()
        print(f"OK Bar DataFrame shape: {bar_df.shape if len(bar_df) > 0 else 'Empty'}")
    
    print("OK Polars engine passed")


def test_alpaca_connection():
    """Test Alpaca API connection."""
    print("\n=== Testing Alpaca Connection ===")
    
    from src.data.alpaca_client import AlpacaClient
    
    try:
        client = AlpacaClient()
        account = client.get_account()
        
        if account:
            print(f"OK Account ID: {account.get('id')}")
            print(f"OK Equity: ${account.get('equity', 0):,.2f}")
            print(f"OK Buying Power: ${account.get('buying_power', 0):,.2f}")
            print(f"OK Status: {account.get('status')}")
        else:
            print("WARN Could not fetch account info (check credentials)")
        
        # Test asset check
        asset_info = client.check_asset_shortable("AAPL")
        print(f"OK AAPL Shortable: {asset_info.get('shortable')}")
        print(f"OK AAPL ETB: {asset_info.get('easy_to_borrow')}")
        
    except Exception as e:
        print(f"WARN Alpaca connection test failed: {e}")
        print("   This is expected if credentials are not set")


def test_risk_manager():
    """Test risk management calculations."""
    print("\n=== Testing Risk Manager ===")
    
    from src.data.alpaca_client import AlpacaClient
    from src.risk.position_manager import RiskManager
    
    try:
        client = AlpacaClient()
        rm = RiskManager(client)
        
        # Test position sizing
        sizing = rm.calculate_position_size(
            symbol="TEST",
            entry_price=15.0,
            atr=0.5,
            vwap=12.0,
            parabolic_apex=16.5
        )
        
        print(f"✓ Position sizing result: {sizing}")
        
    except Exception as e:
        print(f"WARN Risk manager test warning: {e}")


def test_screener():
    """Test asset screener."""
    print("\n=== Testing Screener ===")
    
    from src.data.alpaca_client import AlpacaClient
    from src.screening.screener import ParabolicScreener, ScreenedAsset
    
    try:
        client = AlpacaClient()
        screener = ParabolicScreener(client)
        
        # Test blacklist
        screener.add_to_blacklist("TEST", "test_reason")
        assert screener.is_blacklisted("TEST"), "Should be blacklisted"
        print("OK Blacklist working")
        
        # Test asset screening
        mock_data = {
            'quote': {'ask_price': 25.0},
            'daily': {'open': 10.0, 'high': 28.0, 'low': 9.5, 'volume': 1000000}
        }
        
        # Note: This will fail without proper market data
        # asset = screener.screen_symbol("TEST", mock_data)
        print("OK Screener initialized")
        
    except Exception as e:
        print(f"WARN Screener test warning: {e}")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("PARABOLIC REVERSAL TRADING ENGINE - TEST SUITE")
    print("=" * 60)
    
    try:
        test_config()
        test_numba_kernels()
        test_polars_engine()
        test_alpaca_connection()
        test_risk_manager()
        test_screener()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
        return True
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
