"""Tests for the trade journal backfill script (Agent A4).

Depends on Agent A1's `src.risk.trade_journal` module being present.
"""
from pathlib import Path

import polars as pl
import pytest


def test_backfill_creates_journal_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADE_JOURNAL_DIR", str(tmp_path))
    csv = tmp_path / "fake.csv"
    csv.write_text(
        "symbol,date,gain_pct,trades,pnl,win,loss\n"
        "AMC,2021-06-02,150.0,1,250.0,1,0\n"
        "GME,2021-01-28,300.0,1,-100.0,0,1\n"
    )
    from src.scripts.backfill_trade_journal import backfill_from_csv

    n = backfill_from_csv(csv)
    assert n == 2

    from src.risk.trade_journal import read_trades

    df = read_trades()
    assert df.shape[0] == 2
    assert set(df["symbol"].to_list()) == {"AMC", "GME"}


def test_backfill_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADE_JOURNAL_DIR", str(tmp_path))
    csv = tmp_path / "fake.csv"
    csv.write_text(
        "symbol,date,gain_pct,trades,pnl,win,loss\n"
        "AMC,2021-06-02,150.0,1,250.0,1,0\n"
    )
    from src.scripts.backfill_trade_journal import backfill_from_csv

    backfill_from_csv(csv)
    backfill_from_csv(csv)  # second run should not duplicate

    from src.risk.trade_journal import read_trades

    df = read_trades()
    assert df.shape[0] == 1
