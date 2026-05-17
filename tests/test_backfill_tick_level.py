"""Tests for the tick-level backfill orchestrator.

Uses mocked setups and a mocked TickBacktestEngineV5 — no real backtests run.
Imports from src.scripts.backfill_helpers (Agent A1's module); if A1 hasn't
committed yet, these tests raise ImportError on collection — that's expected.
"""
from __future__ import annotations

# Importing src.scripts.backfill_tick_level transitively triggers
# src/backtest/__init__.py, which eagerly constructs singletons that need
# Alpaca credentials. In CI those env vars don't exist. Set dummy values
# BEFORE the imports below so the singletons construct against a stub.
import os
os.environ.setdefault("ALPACA_API_KEY", "ci-stub-key")
os.environ.setdefault("ALPACA_SECRET", "ci-stub-secret")
os.environ.setdefault("ALPACA_SECRET_KEY", "ci-stub-secret")

import pickle
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest


class _FakeSetup:
    """Minimal duck-typed stand-in for ParabolicSetup in the pickle."""

    def __init__(self, symbol: str, date: str):
        self.symbol = symbol
        self.date = date


def _make_bars_parquet(path: Path, day: datetime) -> None:
    n = 60
    df = pl.DataFrame(
        {
            "timestamp": [day + timedelta(minutes=i) for i in range(n)],
            "open": [10.0] * n,
            "high": [10.2] * n,
            "low": [9.8] * n,
            "close": [10.0] * n,
            "volume": [10000.0] * n,
            "vwap": [10.0] * n,
            "symbol": ["AMC"] * n,
        }
    )
    df.write_parquet(path)


def test_backfill_runs_engine_and_writes_journal(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADE_JOURNAL_DIR", str(tmp_path / "journal"))

    pickle_path = tmp_path / "setups.pkl"
    with pickle_path.open("wb") as f:
        pickle.dump([_FakeSetup("AMC", "2021-06-02")], f)

    bars_dir = tmp_path / "bars"
    bars_dir.mkdir()
    _make_bars_parquet(
        bars_dir / "AMC_1min_20190101_20241231.parquet",
        datetime(2021, 6, 2, 10, 0),
    )

    # Fake engine result: one closed trade
    fake_entry = MagicMock(
        timestamp=datetime(2021, 6, 2, 10, 30),
        price=12.50,
        shares=100,
        add_level=1,
        vwap_extension=0.20,
        volume_ratio=4.2,
        exit_reason=None,
    )
    fake_exit = MagicMock(
        timestamp=datetime(2021, 6, 2, 10, 45),
        price=11.80,
        shares=-100,
        add_level=0,
        exit_reason="tp1",
    )
    fake_result = MagicMock(audit_records=[fake_entry, fake_exit])

    with patch("src.scripts.backfill_tick_level.TickBacktestEngineV5") as MockEngine:
        instance = MockEngine.return_value
        instance.run_tick_backtest.return_value = fake_result
        from src.scripts.backfill_tick_level import backfill

        stats = backfill(pickle_path=pickle_path, bars_dir=bars_dir)

    assert stats["setups_processed"] == 1
    assert stats["trades_written"] == 1

    from src.risk.trade_journal import read_trades

    df = read_trades()
    assert df.shape[0] == 1
    assert df["symbol"].item(0) == "AMC"


def test_backfill_skips_already_journaled_setups(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADE_JOURNAL_DIR", str(tmp_path / "journal"))

    # Seed the journal with a trade for AMC on 2021-06-02
    from src.risk.trade_journal import append_trade

    append_trade(
        {
            "symbol": "AMC",
            "entry_time": datetime(2021, 6, 2, 10, 30),
            "exit_time": datetime(2021, 6, 2, 10, 45),
            "entry_price": 12.5,
            "exit_price": 11.8,
            "shares": 100,
            "side": "short",
            "pnl": 70.0,
            "r_multiple": 1.4,
            "hold_seconds": 900,
            "exit_reason": "tp1",
            "win": True,
            "feat_vwap_extension": 0.2,
            "feat_volume_ratio": 4.2,
            "feat_atr_pct": 0.02,
            "feat_time_of_day_min": 60.0,
            "feat_day_of_week": 2.0,
            "feat_factors_count": 3.0,
        }
    )

    pickle_path = tmp_path / "setups.pkl"
    with pickle_path.open("wb") as f:
        pickle.dump([_FakeSetup("AMC", "2021-06-02")], f)

    with patch("src.scripts.backfill_tick_level.TickBacktestEngineV5") as MockEngine:
        from src.scripts.backfill_tick_level import backfill

        stats = backfill(
            pickle_path=pickle_path,
            bars_dir=tmp_path / "bars",
            skip_existing=True,
        )

    assert stats["setups_processed"] == 0
    assert stats["setups_skipped"] == 1
    MockEngine.return_value.run_tick_backtest.assert_not_called()


def test_backfill_max_setups_caps_work(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADE_JOURNAL_DIR", str(tmp_path / "journal"))
    pickle_path = tmp_path / "setups.pkl"
    setups = [_FakeSetup(f"S{i}", "2021-06-02") for i in range(10)]
    with pickle_path.open("wb") as f:
        pickle.dump(setups, f)

    bars_dir = tmp_path / "bars"
    bars_dir.mkdir()
    for i in range(10):
        _make_bars_parquet(
            bars_dir / f"S{i}_1min_20190101_20241231.parquet",
            datetime(2021, 6, 2, 10, 0),
        )

    with patch("src.scripts.backfill_tick_level.TickBacktestEngineV5") as MockEngine:
        MockEngine.return_value.run_tick_backtest.return_value = MagicMock(
            audit_records=[]
        )
        from src.scripts.backfill_tick_level import backfill

        stats = backfill(
            pickle_path=pickle_path,
            bars_dir=bars_dir,
            max_setups=3,
        )

    assert stats["setups_processed"] == 3
