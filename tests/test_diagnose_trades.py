"""Tests for diagnose_trades. Synthetic trades.jsonl fixtures only."""
import os
os.environ.setdefault("ALPACA_API_KEY", "ci-stub-key")
os.environ.setdefault("ALPACA_SECRET", "ci-stub-secret")
os.environ.setdefault("ALPACA_SECRET_KEY", "ci-stub-secret")

import json
from pathlib import Path


def _write_trades(parent: Path, label: str, trades: list[dict]) -> None:
    d = parent / label
    d.mkdir(parents=True)
    with (d / "trades.jsonl").open("w", encoding="utf-8") as f:
        for t in trades:
            f.write(json.dumps(t) + "\n")


def _trade(symbol: str, pnl: float, mfe: float = 0.0, mae: float = 0.0,
           bars_held: int = 5) -> dict:
    return {
        "fold": 1, "phase": "joint_training",
        "symbol": symbol, "date": "2024-12-01",
        "entry_price": 10.0, "exit_price": 10.0 + pnl / 100,
        "entry_time": "2024-12-01T10:00:00-04:00",
        "exit_time": "2024-12-01T10:05:00-04:00",
        "shares": 100, "pnl": pnl, "return_pct": pnl / 100,
        "win": pnl > 0, "bars_held": bars_held,
        "vwap_at_entry": 9.5, "mfe": mfe, "mae": mae,
    }


def test_load_trades_groups_by_weight(tmp_path):
    from src.scripts.diagnose_trades import _load_trades
    _write_trades(tmp_path, "w0.00_s1", [_trade("X", 10.0), _trade("Y", -5.0)])
    _write_trades(tmp_path, "w0.10_s1", [_trade("Z", 3.0)])
    out = _load_trades(tmp_path)
    assert sorted(out.keys()) == [0.0, 0.1]
    assert len(out[0.0]) == 2
    assert len(out[0.1]) == 1


def test_summarize_basic_stats(tmp_path):
    from src.scripts.diagnose_trades import _summarize_one_weight
    trades = [_trade("X", 100.0), _trade("X", -50.0), _trade("Y", 75.0)]
    s = _summarize_one_weight(trades)
    assert s["n"] == 3
    assert s["total_pnl"] == 125.0
    assert s["win_rate"] == 2 / 3
    assert s["mean_winner_pnl"] == 87.5
    assert s["mean_loser_pnl"] == -50.0
    assert abs(s["expectancy_ratio"] - 87.5 / 50.0) < 1e-6


def test_analyze_smoke(tmp_path):
    from src.scripts.diagnose_trades import analyze
    # 6 winners + 4 losers per weight, two weights
    for w in ["w0.00_s1", "w0.10_s1"]:
        _write_trades(tmp_path, w,
            [_trade(f"SYM{i}", 50.0, mfe=80.0, mae=-5.0, bars_held=3) for i in range(6)]
            + [_trade(f"SYM{i}", -30.0, mfe=10.0, mae=-50.0, bars_held=8) for i in range(4)]
        )
    report = analyze(tmp_path)
    assert "Per-weight summary" in report
    assert "0.00" in report and "0.10" in report
    assert "MFE / MAE" in report
    assert "expectancy_ratio" in report


def test_analyze_flags_negative_expectancy(tmp_path):
    """Synthetic: small winners, big losers -> should call out negative expectancy."""
    from src.scripts.diagnose_trades import analyze
    _write_trades(tmp_path, "w0.00_s1",
        [_trade(f"S{i}", 10.0) for i in range(5)]
        + [_trade(f"S{i}", -50.0) for i in range(5)]
    )
    report = analyze(tmp_path)
    assert "Negative expectancy structure" in report


def test_analyze_handles_empty_sweep(tmp_path):
    from src.scripts.diagnose_trades import analyze
    out = analyze(tmp_path)
    assert "No trades" in out


def test_per_symbol_qualifies_min_5_trades(tmp_path):
    """Symbols with <5 trades should NOT appear in top/bottom lists."""
    from src.scripts.diagnose_trades import _summarize_one_weight
    trades = (
        [_trade("FREQUENT", 100.0)] * 6
        + [_trade("RARE", -1000.0)] * 3  # very bad but only 3 trades -> excluded
    )
    s = _summarize_one_weight(trades)
    symbols_seen = {sym for sym, _, _ in s["worst_symbols"] + s["best_symbols"]}
    assert "FREQUENT" in symbols_seen
    assert "RARE" not in symbols_seen
