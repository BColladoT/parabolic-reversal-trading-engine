"""Pytest tests for the Numba-JIT indicator kernels.

These are pure-numerical tests with no broker / no network dependency,
suitable for CI on Linux runners.
"""
import numpy as np
import pytest

# Skip whole module if numba isn't installed (e.g. minimal CI image).
numba = pytest.importorskip("numba")
from src.indicators.numba_kernels import (  # noqa: E402
    calculate_atr_numba,
    calculate_position_size_numba,
    calculate_vwap_numba,
)


def _bars(n: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    highs = rng.uniform(10, 20, n).astype(np.float64)
    lows = highs - rng.uniform(0.1, 1.0, n).astype(np.float64)
    closes = ((highs + lows) / 2 + rng.uniform(-0.1, 0.1, n)).astype(np.float64)
    volumes = rng.integers(1000, 10000, n).astype(np.float64)
    return highs, lows, closes, volumes


def test_vwap_shape_and_positive():
    highs, lows, closes, volumes = _bars()
    vwap = calculate_vwap_numba(highs, lows, closes, volumes)
    assert vwap.shape == (100,)
    assert np.all(vwap > 0)
    # VWAP must lie within the [min_low, max_high] envelope of the bars seen so far.
    assert vwap[-1] >= lows.min()
    assert vwap[-1] <= highs.max()


def test_atr_shape_and_positive():
    highs, lows, closes, volumes = _bars()
    atr = calculate_atr_numba(highs, lows, closes, period=14)
    assert atr.shape == (100,)
    assert atr[-1] > 0
    # ATR cannot exceed the full price range of any single bar by much.
    assert atr[-1] < (highs.max() - lows.min()) * 2


def test_position_sizing_one_percent_risk():
    """1% of $25K = $250 risk; with $1.50 stop distance -> 166 shares."""
    shares = calculate_position_size_numba(
        account_equity=25_000.0,
        risk_percent=1.0,
        entry_price=15.0,
        stop_price=16.5,
        max_position_value=50_000.0,
    )
    assert shares == 166


def test_position_sizing_capped_by_max_value():
    """If the un-capped sizing exceeds max_position_value, the result must be capped."""
    shares = calculate_position_size_numba(
        account_equity=10_000_000.0,  # huge equity
        risk_percent=1.0,
        entry_price=10.0,
        stop_price=10.10,             # tiny stop -> would be many shares
        max_position_value=30_000.0,  # $30K cap from CLAUDE.md
    )
    assert shares <= int(30_000.0 / 10.0)


def test_position_sizing_zero_when_stop_below_entry_for_short():
    """Short entry needs stop > entry. Pathological inputs must return 0, not crash."""
    shares = calculate_position_size_numba(
        account_equity=25_000.0,
        risk_percent=1.0,
        entry_price=15.0,
        stop_price=15.0,  # zero stop distance
        max_position_value=50_000.0,
    )
    assert shares == 0
