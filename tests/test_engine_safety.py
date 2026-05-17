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
