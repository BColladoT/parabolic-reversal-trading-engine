"""Tests for Position.entry_features, Kelly-aware sizing, and drawdown modulation."""
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Task A3.1 — Position carries entry_features through open_position
# ---------------------------------------------------------------------------
def test_position_has_entry_features_field():
    from src.risk.position_manager import Position
    p = Position(symbol="X")
    assert hasattr(p, "entry_features")
    assert p.entry_features == {}


def test_open_position_accepts_entry_features(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADE_JOURNAL_DIR", str(tmp_path))
    monkeypatch.setenv("DAILY_STATE_PATH", str(tmp_path / "ds.json"))
    from src.risk.position_manager import RiskManager
    rm = RiskManager(alpaca_client=MagicMock())
    p = rm.open_position(
        symbol="AMC", entry_price=10.0, qty=100,
        stop_loss=10.5, vwap=8.0, day_high=11.0, add_level=1,
        entry_features={"vwap_extension": 0.25, "volume_ratio": 3.0},
    )
    assert p.entry_features["vwap_extension"] == 0.25


# ---------------------------------------------------------------------------
# Task A3.3 — Kelly-aware sizing + drawdown modulation in calculate_position_size
# ---------------------------------------------------------------------------
def _seed_journal(tmp_path, monkeypatch, wins: int, losses: int):
    monkeypatch.setenv("TRADE_JOURNAL_DIR", str(tmp_path))
    from src.risk.trade_journal import append_trade
    base = datetime.now() - timedelta(days=1)
    for i in range(wins):
        append_trade({
            "symbol": "X", "entry_time": base + timedelta(minutes=i),
            "exit_time": base + timedelta(minutes=i + 5),
            "entry_price": 10.0, "exit_price": 9.5, "shares": 100, "side": "short",
            "pnl": 50.0, "r_multiple": 1.0, "hold_seconds": 300, "exit_reason": "tp1",
            "win": True, "feat_vwap_extension": 0.2, "feat_volume_ratio": 3.0,
            "feat_atr_pct": 0.02, "feat_time_of_day_min": 60.0,
            "feat_day_of_week": 0.0, "feat_factors_count": 3.0,
        })
    for i in range(losses):
        append_trade({
            "symbol": "X", "entry_time": base + timedelta(minutes=100 + i),
            "exit_time": base + timedelta(minutes=100 + i + 5),
            "entry_price": 10.0, "exit_price": 10.5, "shares": 100, "side": "short",
            "pnl": -50.0, "r_multiple": -1.0, "hold_seconds": 300, "exit_reason": "stop",
            "win": False, "feat_vwap_extension": 0.2, "feat_volume_ratio": 3.0,
            "feat_atr_pct": 0.02, "feat_time_of_day_min": 60.0,
            "feat_day_of_week": 0.0, "feat_factors_count": 3.0,
        })


def test_calculate_position_size_uses_kelly_when_journal_has_data(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADE_JOURNAL_DIR", str(tmp_path))
    monkeypatch.setenv("DAILY_STATE_PATH", str(tmp_path / "ds.json"))
    _seed_journal(tmp_path, monkeypatch, wins=60, losses=40)
    from src.risk.position_manager import RiskManager
    rm = RiskManager(alpaca_client=MagicMock())
    # Bypass update_account()'s broker call by stubbing client.get_account
    rm.client.get_account = MagicMock(return_value={"equity": 100_000, "buying_power": 500_000})
    rm.account_equity = 100_000
    sizing = rm.calculate_position_size(
        symbol="X", entry_price=10.0, atr=0.20, vwap=8.0, day_high=11.0, add_level=1
    )
    assert sizing["valid"] is True
    # With ~60% win rate, expected R = 0.2 -> kelly fraction = 0.2 (under quarter-Kelly cap)
    assert sizing.get("kelly_fraction", 0.0) > 0.0
    assert sizing.get("edge_n_trades", 0) == 100


def test_calculate_position_size_falls_back_to_fixed_with_empty_journal(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADE_JOURNAL_DIR", str(tmp_path))
    monkeypatch.setenv("DAILY_STATE_PATH", str(tmp_path / "ds.json"))
    from src.risk.position_manager import RiskManager
    rm = RiskManager(alpaca_client=MagicMock())
    rm.client.get_account = MagicMock(return_value={"equity": 100_000, "buying_power": 500_000})
    rm.account_equity = 100_000
    sizing = rm.calculate_position_size(
        symbol="X", entry_price=10.0, atr=0.20, vwap=8.0, day_high=11.0, add_level=1
    )
    assert sizing["valid"] is True
    # Empty journal: kelly_fraction should be 0; sizing falls back to fixed-% logic
    assert sizing.get("kelly_fraction", 0.0) == 0.0
    assert sizing.get("edge_n_trades", 0) == 0


def test_calculate_position_size_halves_after_three_consecutive_losses(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADE_JOURNAL_DIR", str(tmp_path))
    monkeypatch.setenv("DAILY_STATE_PATH", str(tmp_path / "ds.json"))
    _seed_journal(tmp_path, monkeypatch, wins=20, losses=0)  # baseline winning history
    # Then 3 losses at the tail
    from src.risk.trade_journal import append_trade
    base = datetime.now()
    for i in range(3):
        append_trade({
            "symbol": "X", "entry_time": base + timedelta(minutes=i),
            "exit_time": base + timedelta(minutes=i + 1),
            "entry_price": 10.0, "exit_price": 10.5, "shares": 100, "side": "short",
            "pnl": -50.0, "r_multiple": -1.0, "hold_seconds": 60, "exit_reason": "stop",
            "win": False, "feat_vwap_extension": 0.2, "feat_volume_ratio": 3.0,
            "feat_atr_pct": 0.02, "feat_time_of_day_min": 60.0,
            "feat_day_of_week": 0.0, "feat_factors_count": 3.0,
        })
    from src.risk.position_manager import RiskManager
    rm = RiskManager(alpaca_client=MagicMock())
    rm.client.get_account = MagicMock(return_value={"equity": 100_000, "buying_power": 500_000})
    rm.account_equity = 100_000

    sizing_a = rm.calculate_position_size(
        symbol="X", entry_price=10.0, atr=0.20, vwap=8.0, day_high=11.0, add_level=1
    )
    assert sizing_a.get("dd_modifier", 1.0) <= 0.5  # halved (or quartered) due to 3 losses
