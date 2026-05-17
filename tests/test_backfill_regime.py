"""Tests for src/scripts/backfill_regime.py.

yfinance is mocked via patching ``src.scripts.backfill_regime.fetch_regime_history``
so the CLI logic is exercised without any network call.
"""
from datetime import date
from unittest.mock import patch

import polars as pl


def test_backfill_writes_regime_table(tmp_path, monkeypatch):
    monkeypatch.setenv("REGIME_DIR", str(tmp_path))
    from src.risk.regime import REGIME_SCHEMA, read_regime_history
    fake_df = pl.DataFrame({
        "date": [date(2024, 1, 2), date(2024, 1, 3)],
        "vix_level": [15.0, 25.0],
        "spy_trend": [1, -1],
        "label": ["risk_on", "risk_off"],
    }, schema=REGIME_SCHEMA)
    with patch("src.scripts.backfill_regime.fetch_regime_history", return_value=fake_df):
        from src.scripts.backfill_regime import backfill
        n = backfill(start=date(2024, 1, 1), end=date(2024, 1, 10))
    assert n == 2
    assert read_regime_history().shape[0] == 2


def test_backfill_empty_yfinance_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("REGIME_DIR", str(tmp_path))
    from src.risk.regime import REGIME_SCHEMA, read_regime_history
    empty = pl.DataFrame(schema=REGIME_SCHEMA)
    with patch("src.scripts.backfill_regime.fetch_regime_history", return_value=empty):
        from src.scripts.backfill_regime import backfill
        n = backfill(start=date(2024, 1, 1), end=date(2024, 1, 10))
    assert n == 0
    # No parquet written -> read returns empty schema-only frame
    assert read_regime_history().is_empty()


def test_backfill_cli_main_prints_summary(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("REGIME_DIR", str(tmp_path))
    from src.risk.regime import REGIME_SCHEMA
    fake_df = pl.DataFrame({
        "date": [date(2024, 1, 2)],
        "vix_level": [15.0],
        "spy_trend": [1],
        "label": ["risk_on"],
    }, schema=REGIME_SCHEMA)
    import sys
    argv = sys.argv
    try:
        sys.argv = ["backfill_regime", "--start", "2024-01-01", "--end", "2024-01-10"]
        with patch("src.scripts.backfill_regime.fetch_regime_history", return_value=fake_df):
            from src.scripts.backfill_regime import main
            main()
    finally:
        sys.argv = argv
    captured = capsys.readouterr()
    assert "1" in captured.out
    assert "regime" in captured.out.lower()
