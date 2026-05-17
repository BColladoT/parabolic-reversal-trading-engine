from datetime import datetime, date
from pathlib import Path
import polars as pl
import pytest


def _sample_record() -> dict:
    return {
        "symbol": "AMC",
        "entry_time": datetime(2026, 5, 17, 10, 30),
        "exit_time": datetime(2026, 5, 17, 10, 45),
        "entry_price": 12.50,
        "exit_price": 11.80,
        "shares": 100,
        "side": "short",
        "pnl": 70.0,
        "r_multiple": 1.4,
        "hold_seconds": 900,
        "exit_reason": "tp1",
        "win": True,
        "feat_vwap_extension": 0.18,
        "feat_volume_ratio": 4.2,
        "feat_atr_pct": 0.022,
        "feat_time_of_day_min": 60.0,
        "feat_day_of_week": 0.0,
        "feat_factors_count": 3.0,
    }


def test_append_creates_partition_file(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADE_JOURNAL_DIR", str(tmp_path))
    from src.risk.trade_journal import append_trade
    p = append_trade(_sample_record())
    assert p.exists()
    assert p.name == "2026-05-17.parquet"


def test_append_then_read_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADE_JOURNAL_DIR", str(tmp_path))
    from src.risk.trade_journal import append_trade, read_trades
    append_trade(_sample_record())
    df = read_trades()
    assert df.shape[0] == 1
    assert df["symbol"].item(0) == "AMC"
    assert df["pnl"].item(0) == pytest.approx(70.0)


def test_append_concurrent_writes_same_day(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADE_JOURNAL_DIR", str(tmp_path))
    from src.risk.trade_journal import append_trade, read_trades
    for i in range(5):
        rec = _sample_record()
        rec["symbol"] = f"SYM{i}"
        append_trade(rec)
    df = read_trades()
    assert df.shape[0] == 5


def test_read_trades_date_filter(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADE_JOURNAL_DIR", str(tmp_path))
    from src.risk.trade_journal import append_trade, read_trades
    r1 = _sample_record()
    r2 = _sample_record()
    r2["entry_time"] = datetime(2026, 5, 18, 10, 30)
    append_trade(r1)
    append_trade(r2)
    df = read_trades(date_from=date(2026, 5, 18))
    assert df.shape[0] == 1
    assert df["entry_time"].item(0).date() == date(2026, 5, 18)


def test_read_trades_empty_when_no_files(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADE_JOURNAL_DIR", str(tmp_path))
    from src.risk.trade_journal import read_trades
    df = read_trades()
    assert df.is_empty()
