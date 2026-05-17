"""Tests for src/risk/regime.py and the compute_edge(regime_label=...) filter.

Uses tmp_path + monkeypatch.setenv to isolate REGIME_DIR / TRADE_JOURNAL_DIR
per test. yfinance is mocked via unittest.mock.patch — no network in tests.
"""
from datetime import date, datetime, timedelta
from unittest.mock import patch

import polars as pl
import pytest


def _fake_vix_df():
    import pandas as pd
    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    return pd.DataFrame({"Close": [15.0, 18.0, 22.0, 28.0, 30.0]}, index=idx)


def _fake_spy_df():
    import pandas as pd
    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    # SPY closes above the 50d SMA so spy_trend >= 0
    closes = [510.0, 515.0, 505.0, 500.0, 495.0]
    return pd.DataFrame({"Close": closes}, index=idx)


def test_regime_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("REGIME_DIR", str(tmp_path))
    from src.risk.regime import write_regime_history, read_regime_history, REGIME_SCHEMA
    df = pl.DataFrame({
        "date": [date(2024, 1, 2), date(2024, 1, 3)],
        "vix_level": [15.0, 18.0],
        "spy_trend": [1, 0],
        "label": ["risk_on", "neutral"],
    }, schema=REGIME_SCHEMA)
    write_regime_history(df)
    out = read_regime_history()
    assert out.shape[0] == 2
    assert set(out.columns) == set(REGIME_SCHEMA.keys())


def test_regime_for_date_returns_row(tmp_path, monkeypatch):
    monkeypatch.setenv("REGIME_DIR", str(tmp_path))
    from src.risk.regime import write_regime_history, regime_for_date, REGIME_SCHEMA
    write_regime_history(pl.DataFrame({
        "date": [date(2024, 1, 2)],
        "vix_level": [15.0],
        "spy_trend": [1],
        "label": ["risk_on"],
    }, schema=REGIME_SCHEMA))
    r = regime_for_date(date(2024, 1, 2))
    assert r is not None
    assert r.label == "risk_on"
    assert r.vix_level == 15.0
    assert r.spy_trend == 1


def test_regime_for_date_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("REGIME_DIR", str(tmp_path))
    from src.risk.regime import regime_for_date
    assert regime_for_date(date(2099, 1, 1)) is None


def test_read_regime_history_empty_when_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("REGIME_DIR", str(tmp_path))
    from src.risk.regime import read_regime_history, REGIME_SCHEMA
    df = read_regime_history()
    assert df.is_empty()
    assert set(df.columns) == set(REGIME_SCHEMA.keys())


def test_fetch_regime_history_labels(tmp_path, monkeypatch):
    monkeypatch.setenv("REGIME_DIR", str(tmp_path))
    with patch("src.risk.regime._download_yf") as dl:
        # vix first, then spy
        dl.side_effect = [_fake_vix_df(), _fake_spy_df()]
        from src.risk.regime import fetch_regime_history
        df = fetch_regime_history(date(2024, 1, 1), date(2024, 1, 10))
    assert df.shape[0] == 5
    # vix=15 + spy_trend>=0 -> risk_on
    assert df["label"].item(0) == "risk_on"
    # vix=30 -> risk_off (>=25)
    assert df["label"].item(4) == "risk_off"
    # vix_level column should be float
    assert df["vix_level"].item(0) == pytest.approx(15.0)


def test_fetch_regime_history_empty_when_yf_returns_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("REGIME_DIR", str(tmp_path))
    import pandas as pd
    empty = pd.DataFrame({"Close": []})
    with patch("src.risk.regime._download_yf") as dl:
        dl.side_effect = [empty, empty]
        from src.risk.regime import fetch_regime_history
        df = fetch_regime_history(date(2024, 1, 1), date(2024, 1, 10))
    assert df.is_empty()


# ----------------------------------------------------------------------
# Task A2.3: compute_edge(regime_label=...) integration tests
# ----------------------------------------------------------------------


def _make_trade(t: datetime, win: bool) -> dict:
    return {
        "symbol": "X",
        "entry_time": t,
        "exit_time": t + timedelta(minutes=5),
        "entry_price": 10.0,
        "exit_price": 9.5,
        "shares": 100,
        "side": "short",
        "pnl": 50.0 if win else -50.0,
        "r_multiple": 1.0 if win else -1.0,
        "hold_seconds": 300,
        "exit_reason": "tp1",
        "win": win,
        "feat_vwap_extension": 0.2,
        "feat_volume_ratio": 3.0,
        "feat_atr_pct": 0.02,
        "feat_time_of_day_min": 60.0,
        "feat_day_of_week": 0.0,
        "feat_factors_count": 3.0,
    }


def test_compute_edge_filters_by_regime(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADE_JOURNAL_DIR", str(tmp_path / "journal"))
    monkeypatch.setenv("REGIME_DIR", str(tmp_path / "regime"))
    from src.risk.trade_journal import append_trade
    from src.risk.regime import write_regime_history, REGIME_SCHEMA

    # 20 trades on 2024-01-02 (risk_on): 12 wins, 8 losses -> 60% wr
    base = datetime(2024, 1, 2, 10, 30)
    for i in range(20):
        append_trade(_make_trade(base + timedelta(minutes=i), win=i < 12))
    # 20 trades on 2024-06-02 (risk_off): 2 wins, 18 losses -> 10% wr
    base_off = datetime(2024, 6, 2, 10, 30)
    for i in range(20):
        append_trade(_make_trade(base_off + timedelta(minutes=i), win=i < 2))

    write_regime_history(pl.DataFrame({
        "date": [date(2024, 1, 2), date(2024, 6, 2)],
        "vix_level": [15.0, 30.0],
        "spy_trend": [1, -1],
        "label": ["risk_on", "risk_off"],
    }, schema=REGIME_SCHEMA))

    from src.risk.edge_estimator import compute_edge

    on = compute_edge(lookback_days=None, regime_label="risk_on", min_samples_conditional=15)
    off = compute_edge(lookback_days=None, regime_label="risk_off", min_samples_conditional=15)
    assert on.win_rate > 0.5
    assert off.win_rate < 0.2
    assert on.used_fallback is False
    assert off.used_fallback is False
    assert on.n_trades == 20
    assert off.n_trades == 20


def test_compute_edge_regime_no_table_falls_through(tmp_path, monkeypatch):
    """If regime table is missing, compute_edge should not crash; it falls back
    to the un-regime-filtered trade set."""
    monkeypatch.setenv("TRADE_JOURNAL_DIR", str(tmp_path / "journal"))
    monkeypatch.setenv("REGIME_DIR", str(tmp_path / "regime"))  # empty dir = no parquet
    from src.risk.trade_journal import append_trade
    base = datetime(2024, 1, 2, 10, 30)
    for i in range(10):
        append_trade(_make_trade(base + timedelta(minutes=i), win=i < 6))
    from src.risk.edge_estimator import compute_edge
    e = compute_edge(lookback_days=None, regime_label="risk_on")
    # No regime table -> filter is a no-op -> returns overall stats over all 10 trades
    assert e.n_trades == 10


def test_inspect_journal_regime_flag_smoke(tmp_path, monkeypatch, capsys):
    """The --regime flag should thread through to _print_overall and not crash
    even when the journal is empty."""
    monkeypatch.setenv("TRADE_JOURNAL_DIR", str(tmp_path / "journal"))
    monkeypatch.setenv("REGIME_DIR", str(tmp_path / "regime"))
    from src.scripts import inspect_journal
    inspect_journal._print_overall(regime_label="risk_on")
    captured = capsys.readouterr()
    assert "risk_on" in captured.out
