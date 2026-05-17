"""Tests for journal persistence triggered by RiskManager.close_position."""
from datetime import datetime
from unittest.mock import MagicMock


def _open_short_position(rm, tmp_path):
    rm.open_position(
        symbol="AMC", entry_price=10.0, qty=100,
        stop_loss=10.5, vwap=8.0, day_high=11.0, add_level=1,
        entry_features={
            "vwap_extension": 0.25, "volume_ratio": 3.0, "atr_pct": 0.02,
            "time_of_day_min": 60.0, "day_of_week": 0.0, "factors_count": 3.0,
        },
    )


def test_close_position_writes_journal_row(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADE_JOURNAL_DIR", str(tmp_path))
    monkeypatch.setenv("DAILY_STATE_PATH", str(tmp_path / "ds.json"))
    from src.risk.position_manager import RiskManager
    from src.risk.trade_journal import read_trades
    rm = RiskManager(alpaca_client=MagicMock())
    _open_short_position(rm, tmp_path)
    rm.close_position(symbol="AMC", exit_price=9.5, reason="tp1")

    df = read_trades()
    assert df.shape[0] == 1
    row = df.row(0, named=True)
    assert row["symbol"] == "AMC"
    assert row["pnl"] > 0  # short closed lower
    assert row["win"] is True
    assert row["exit_reason"] == "tp1"
    assert row["feat_vwap_extension"] == 0.25


def test_close_position_handles_missing_journal_gracefully(tmp_path, monkeypatch):
    # If the journal write fails internally, close_position must not raise.
    # We force a failure by replacing the journal dir env with a path that
    # cannot be written to after open. Simplest approach: monkeypatch
    # append_trade itself to raise; the close should still return pnl.
    monkeypatch.setenv("TRADE_JOURNAL_DIR", str(tmp_path / "nope"))
    monkeypatch.setenv("DAILY_STATE_PATH", str(tmp_path / "ds.json"))
    from src.risk import position_manager as pm
    from src.risk.position_manager import RiskManager

    def _boom(_record):
        raise IOError("simulated journal failure")

    monkeypatch.setattr(pm, "append_trade", _boom)

    rm = RiskManager(alpaca_client=MagicMock())
    rm.open_position("AMC", 10.0, 100, 10.5, 8.0, 11.0, 1, entry_features={})
    # Should not raise even if journal write fails
    pnl = rm.close_position("AMC", 9.5, "tp1")
    assert pnl > 0
