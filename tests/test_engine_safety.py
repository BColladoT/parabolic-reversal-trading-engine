"""Agent 1 — Live Engine Safety tests.

Hardening tests for circuit breaker enforcement, daily reset, state
persistence, broker reconciliation on startup, and typed exception handling.
"""
from unittest.mock import MagicMock, patch
import pytest

from src.risk.position_manager import RiskManager


def test_check_daily_loss_limit_blocks_when_hit():
    rm = RiskManager(alpaca_client=MagicMock())
    rm.account_equity = 100_000
    rm.daily_pnl = -25_000  # blow through any reasonable limit
    assert rm.check_daily_loss_limit() is True
    assert rm.daily_stats['daily_loss_limit_hit'] is True


def test_check_daily_loss_limit_sticky_once_tripped():
    rm = RiskManager(alpaca_client=MagicMock())
    rm.account_equity = 100_000
    rm.daily_pnl = -25_000
    rm.check_daily_loss_limit()
    rm.daily_pnl = -100  # recover
    assert rm.check_daily_loss_limit() is True  # still locked


def test_execute_entry_aborts_when_daily_limit_hit():
    from src.main_engine import TradingEngine
    eng = TradingEngine.__new__(TradingEngine)  # skip __init__
    eng.risk_manager = MagicMock()
    eng.risk_manager.check_daily_loss_limit.return_value = True
    eng.risk_manager.positions = {}
    eng.risk_manager.daily_pnl = -25_000
    eng.data_engine = MagicMock()
    eng.screener = MagicMock(screened_assets={})
    eng.alpaca = MagicMock()

    signal = MagicMock(symbol="AMC", price=10.0, atr=0.5, vwap=8.0)
    eng._execute_entry(signal)

    eng.alpaca.submit_short_order.assert_not_called()
    eng.risk_manager.calculate_position_size.assert_not_called()


def test_reset_daily_stats_fires_on_new_et_day():
    from src.main_engine import TradingEngine
    eng = TradingEngine.__new__(TradingEngine)
    eng.risk_manager = MagicMock()
    eng.error_count = 5
    eng._last_reset_date = None

    fake_now = MagicMock()
    fake_now.date.return_value = "2026-05-18"
    eng._maybe_reset_daily(fake_now)
    eng.risk_manager.reset_daily_stats.assert_called_once()
    assert eng.error_count == 0

    eng.risk_manager.reset_daily_stats.reset_mock()
    eng._maybe_reset_daily(fake_now)  # same day -> no second call
    eng.risk_manager.reset_daily_stats.assert_not_called()


def test_daily_state_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("DAILY_STATE_PATH", str(tmp_path / "daily_state.json"))
    rm1 = RiskManager(alpaca_client=MagicMock())
    rm1.account_equity = 100_000
    rm1.daily_pnl = -1234.5
    rm1._persist_daily_state()

    rm2 = RiskManager(alpaca_client=MagicMock())
    rm2._restore_daily_state()
    assert rm2.daily_pnl == pytest.approx(-1234.5)


def test_daily_state_ignored_when_stale_date(tmp_path, monkeypatch):
    """State from yesterday should NOT be restored (we want a fresh slate)."""
    import json
    state_file = tmp_path / "daily_state.json"
    state_file.write_text(json.dumps({
        "date": "1999-01-01",
        "daily_pnl": -9999.0,
        "daily_loss_limit_hit": True,
    }))
    monkeypatch.setenv("DAILY_STATE_PATH", str(state_file))

    rm = RiskManager(alpaca_client=MagicMock())
    rm._restore_daily_state()
    assert rm.daily_pnl == 0.0
    assert rm.daily_stats['daily_loss_limit_hit'] is False
