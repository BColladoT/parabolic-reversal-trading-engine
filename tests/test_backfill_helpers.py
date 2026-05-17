"""Tests for src.scripts.backfill_helpers — pure adapter helpers."""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import polars as pl
import pytest


# --- Shared fixtures (module-scope so all task tests can reuse) ----------

def _make_bars(start: datetime, n: int = 20, true_range: float = 0.10) -> pl.DataFrame:
    """Build a deterministic 1-min OHLC frame for ATR tests."""
    return pl.DataFrame({
        "timestamp": [start + timedelta(minutes=i) for i in range(n)],
        "open": [10.0] * n,
        "high": [10.0 + true_range] * n,
        "low": [10.0 - true_range] * n,
        "close": [10.0] * n,
    })


def _mock_engine_result_with_one_trade():
    entry = MagicMock(
        timestamp=datetime(2021, 6, 2, 10, 30),
        price=12.50,
        shares=100,
        add_level=1,
        vwap_extension=0.20,
        volume_ratio=4.2,
        confirming_factors=3,
        exit_reason=None,
    )
    exit_ = MagicMock(
        timestamp=datetime(2021, 6, 2, 10, 45),
        price=11.80,
        shares=-100,
        add_level=0,
        exit_reason="tp1",
    )
    result = MagicMock(audit_records=[entry, exit_])
    return result


# --- Task A1.1: BacktestTrade pairing ------------------------------------

def test_trades_from_engine_result_pairs_entry_exit():
    from src.scripts.backfill_helpers import trades_from_engine_result
    trades = trades_from_engine_result(_mock_engine_result_with_one_trade(), "AMC")
    assert len(trades) == 1
    t = trades[0]
    assert t.entry_price == 12.50
    assert t.exit_price == 11.80
    assert t.shares == 100
    assert t.exit_reason == "tp1"
    assert t.pnl == pytest.approx((12.50 - 11.80) * 100)  # short: entry - exit
    assert t.vwap_extension == 0.20


def test_trades_from_engine_result_empty_when_no_audit():
    from src.scripts.backfill_helpers import trades_from_engine_result
    result = MagicMock(audit_records=[])
    assert trades_from_engine_result(result, "X") == []


def test_trades_from_engine_result_handles_multi_leg_scale_in():
    """Engine may emit entry -> add -> exit. Treat as one weighted-average trade."""
    e1 = MagicMock(timestamp=datetime(2021, 6, 2, 10, 30), price=12.00, shares=50,
                   add_level=1, vwap_extension=0.18, volume_ratio=3.0,
                   confirming_factors=2, exit_reason=None)
    e2 = MagicMock(timestamp=datetime(2021, 6, 2, 10, 35), price=12.50, shares=50,
                   add_level=2, vwap_extension=0.22, volume_ratio=3.5,
                   confirming_factors=3, exit_reason=None)
    x = MagicMock(timestamp=datetime(2021, 6, 2, 10, 45), price=11.80, shares=-100,
                  add_level=0, exit_reason="tp1")
    from src.scripts.backfill_helpers import trades_from_engine_result
    trades = trades_from_engine_result(MagicMock(audit_records=[e1, e2, x]), "AMC")
    assert len(trades) == 1
    t = trades[0]
    assert t.shares == 100
    # weighted avg entry: (50*12 + 50*12.5) / 100 = 12.25
    assert t.entry_price == pytest.approx(12.25)
    assert t.entry_time == datetime(2021, 6, 2, 10, 30)  # earliest entry
    # vwap_extension uses the FIRST entry's value (signal of initial setup quality)
    assert t.vwap_extension == 0.18


# --- Task A1.2: ATR over 1-min bars --------------------------------------

def test_atr_pct_at_basic():
    from src.scripts.backfill_helpers import atr_pct_at
    start = datetime(2021, 6, 2, 10, 0)
    bars = _make_bars(start, n=20, true_range=0.10)
    target = start + timedelta(minutes=15)
    atr = atr_pct_at(bars, target, window=14)
    # high-low = 0.20 always, close=10.0 -> atr_pct = 0.20 / 10.0 = 0.02
    assert atr == pytest.approx(0.02, abs=1e-3)


def test_atr_pct_at_returns_zero_when_insufficient_bars():
    from src.scripts.backfill_helpers import atr_pct_at
    start = datetime(2021, 6, 2, 10, 0)
    bars = _make_bars(start, n=3)
    target = start + timedelta(minutes=2)
    assert atr_pct_at(bars, target, window=14) == 0.0


def test_atr_pct_at_empty_frame_returns_zero():
    from src.scripts.backfill_helpers import atr_pct_at
    empty = pl.DataFrame(
        schema={"timestamp": pl.Datetime, "open": pl.Float64,
                "high": pl.Float64, "low": pl.Float64, "close": pl.Float64}
    )
    assert atr_pct_at(empty, datetime(2021, 6, 2, 10, 0), window=14) == 0.0
