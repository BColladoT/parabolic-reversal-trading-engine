"""Verifies send_alert fires at each critical safety path in main_engine.

Patched against `src.main_engine.send_alert` (the import-bound name) so the
real urllib.request call is never made.
"""
from unittest.mock import MagicMock, patch
import pytest


@patch("src.main_engine.send_alert")
def test_alert_fires_when_circuit_breaker_trips(mock_alert):
    from src.main_engine import TradingEngine
    eng = TradingEngine.__new__(TradingEngine)
    eng.risk_manager = MagicMock()
    eng.risk_manager.check_daily_loss_limit.return_value = True
    eng.risk_manager.positions = {}
    eng.risk_manager.daily_pnl = -25_000
    eng.data_engine = MagicMock()
    eng.screener = MagicMock(screened_assets={})
    eng.alpaca = MagicMock()

    signal = MagicMock(symbol="AMC", price=10.0, atr=0.5, vwap=8.0)
    eng._execute_entry(signal)

    mock_alert.assert_called_once()
    title = mock_alert.call_args.args[0]
    assert "daily loss" in title.lower()


@patch("src.main_engine.send_alert")
def test_alert_fires_when_startup_blocked_by_broker(mock_alert):
    from src.main_engine import TradingEngine
    eng = TradingEngine.__new__(TradingEngine)
    eng.alpaca = MagicMock()
    eng.alpaca.get_positions.return_value = [{"symbol": "AMC", "qty": -100}]
    eng.risk_manager = MagicMock(positions={})

    with pytest.raises(RuntimeError):
        eng._reconcile_on_startup()

    mock_alert.assert_called_once()
    title = mock_alert.call_args.args[0]
    assert "startup" in title.lower()


@patch("src.main_engine.send_alert")
def test_alert_fires_on_emergency_shutdown(mock_alert):
    from src.main_engine import TradingEngine
    eng = TradingEngine.__new__(TradingEngine)
    eng.risk_manager = MagicMock()
    eng.alpaca = MagicMock()
    eng.running = True

    eng.emergency_shutdown()

    mock_alert.assert_called()
    title = mock_alert.call_args_list[0].args[0]
    assert "emergency" in title.lower()


@patch("src.main_engine.send_alert")
def test_alert_fires_on_stale_feed(mock_alert):
    from src.main_engine import TradingEngine
    eng = TradingEngine.__new__(TradingEngine)
    eng.alpaca = MagicMock()
    eng.alpaca.is_feed_stale.return_value = True
    eng.alpaca.is_connected.return_value = True
    eng.alpaca.get_account.return_value = {"equity": 10_000_000}
    eng.risk_manager = MagicMock()
    eng.risk_manager.get_position_summary.return_value = {
        "open_count": 0, "unrealized_pnl": 0.0, "daily_pnl": 0.0,
    }
    eng._last_stale_alert_ts = 0.0
    eng._stale_alert_min_interval = 300

    eng._health_check()

    assert any("stale" in c.args[0].lower() for c in mock_alert.call_args_list)


@patch("src.main_engine.send_alert")
def test_stale_alert_is_debounced_within_window(mock_alert):
    from src.main_engine import TradingEngine
    eng = TradingEngine.__new__(TradingEngine)
    eng.alpaca = MagicMock()
    eng.alpaca.is_feed_stale.return_value = True
    eng.alpaca.is_connected.return_value = True
    eng.alpaca.get_account.return_value = {"equity": 10_000_000}
    eng.risk_manager = MagicMock()
    eng.risk_manager.get_position_summary.return_value = {
        "open_count": 0, "unrealized_pnl": 0.0, "daily_pnl": 0.0,
    }
    eng._last_stale_alert_ts = 0.0
    eng._stale_alert_min_interval = 300

    eng._health_check()
    eng._health_check()  # immediate retry — should not realert

    stale_calls = [c for c in mock_alert.call_args_list if "stale" in c.args[0].lower()]
    assert len(stale_calls) == 1
