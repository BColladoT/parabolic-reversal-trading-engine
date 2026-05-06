"""Smoke test: the quickstart example script runs end-to-end on sample data."""
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample"


def test_sample_data_present():
    """The five committed sample parquets must exist and be readable."""
    expected = {"GME.parquet", "BBIG.parquet", "KOSS.parquet", "MULN.parquet", "GFAI.parquet"}
    found = {p.name for p in SAMPLE.glob("*.parquet")}
    assert expected <= found, f"Missing sample files: {expected - found}"


def test_sample_data_schema():
    """All sample parquets must have the OHLCV schema the example script expects."""
    required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
    for path in SAMPLE.glob("*.parquet"):
        df = pl.read_parquet(path)
        assert required_cols <= set(df.columns), f"{path.name} missing cols: {required_cols - set(df.columns)}"
        assert len(df) > 0, f"{path.name} is empty"


def test_example_backtest_runs(monkeypatch, tmp_path):
    """The example backtest must import, run, and produce at least one trade."""
    import importlib.util

    monkeypatch.chdir(ROOT)
    script = ROOT / "scripts" / "run_example_backtest.py"
    spec = importlib.util.spec_from_file_location("run_example_backtest", script)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(module, "OUT_IMG", tmp_path / "equity.png", raising=False)
    spec.loader.exec_module(module)
    # Override OUT_IMG after module load so the test does not litter docs/images.
    module.OUT_IMG = tmp_path / "equity.png"

    trades = module.run()
    assert isinstance(trades, list)
    assert len(trades) >= 1, "Example backtest produced zero trades on sample data"
    for t in trades:
        assert {"symbol", "date", "entry_price", "exit_price", "pnl"} <= set(t.keys())
        assert t["entry_price"] > 0
        assert isinstance(t["pnl"], (int, float))


def test_summarize_writes_chart(tmp_path, monkeypatch):
    """summarize() should write the PNG to OUT_IMG.

    OUT_IMG must be inside ROOT because summarize() calls .relative_to(ROOT)
    when printing the path. Use a tmp dir under the repo root.
    """
    import importlib.util

    monkeypatch.chdir(ROOT)
    script = ROOT / "scripts" / "run_example_backtest.py"
    spec = importlib.util.spec_from_file_location("run_example_backtest", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    chart_dir = ROOT / "docs" / "images" / "_pytest_tmp"
    chart_dir.mkdir(parents=True, exist_ok=True)
    chart_path = chart_dir / "equity.png"
    if chart_path.exists():
        chart_path.unlink()
    module.OUT_IMG = chart_path

    try:
        trades = module.run()
        module.summarize(trades)
        assert chart_path.exists()
        assert chart_path.stat().st_size > 0
    finally:
        if chart_path.exists():
            chart_path.unlink()
        try:
            chart_dir.rmdir()
        except OSError:
            pass
