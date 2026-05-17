import math
import numpy as np
import pytest


def test_bootstrap_ci_returns_nan_when_too_few():
    from src.risk.edge_estimator import bootstrap_ci
    low, high = bootstrap_ci([1.0, 0.0, 1.0], statistic=np.mean)
    assert math.isnan(low) and math.isnan(high)


def test_bootstrap_ci_brackets_true_mean_for_balanced_sample():
    from src.risk.edge_estimator import bootstrap_ci
    rng = np.random.default_rng(7)
    sample = rng.binomial(1, 0.7, size=100).astype(float)
    low, high = bootstrap_ci(sample, statistic=np.mean, n_iter=1000, seed=42)
    assert low < 0.7 < high
    assert (high - low) < 0.25


def test_bootstrap_ci_seed_is_reproducible():
    from src.risk.edge_estimator import bootstrap_ci
    sample = [0.0, 1.0, 1.0, 0.0, 1.0] * 20
    a = bootstrap_ci(sample, statistic=np.mean, n_iter=500, seed=123)
    b = bootstrap_ci(sample, statistic=np.mean, n_iter=500, seed=123)
    assert a == b


def test_compute_edge_populates_ci_fields_on_dataclass():
    from src.risk.edge_estimator import EdgeStats
    assert "win_rate_ci_low" in EdgeStats.__dataclass_fields__
    assert "win_rate_ci_high" in EdgeStats.__dataclass_fields__
    assert "expected_r_ci_low" in EdgeStats.__dataclass_fields__
    assert "expected_r_ci_high" in EdgeStats.__dataclass_fields__


def test_compute_edge_ci_brackets_point_estimate(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADE_JOURNAL_DIR", str(tmp_path))
    from datetime import datetime, timedelta
    from src.risk.trade_journal import append_trade
    from src.risk.edge_estimator import compute_edge

    base = datetime(2024, 6, 1, 10, 0)
    for i in range(50):
        is_win = i < 35
        append_trade({
            "symbol": "X", "entry_time": base + timedelta(minutes=i),
            "exit_time": base + timedelta(minutes=i + 5),
            "entry_price": 10.0, "exit_price": 9.5, "shares": 100, "side": "short",
            "pnl": 50.0 if is_win else -50.0,
            "r_multiple": 1.0 if is_win else -1.0,
            "hold_seconds": 300, "exit_reason": "tp1", "win": is_win,
            "feat_vwap_extension": 0.2, "feat_volume_ratio": 3.0, "feat_atr_pct": 0.02,
            "feat_time_of_day_min": 60.0, "feat_day_of_week": 0.0, "feat_factors_count": 3.0,
        })
    e = compute_edge(lookback_days=None)
    assert not math.isnan(e.win_rate_ci_low)
    assert e.win_rate_ci_low < e.win_rate < e.win_rate_ci_high
    assert 0.02 < (e.win_rate_ci_high - e.win_rate_ci_low) < 0.30
