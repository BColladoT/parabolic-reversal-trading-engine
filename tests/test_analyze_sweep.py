"""Tests for the sweep analyzer. Synthetic JSON fixtures, no torch."""
import os
os.environ.setdefault("ALPACA_API_KEY", "ci-stub-key")
os.environ.setdefault("ALPACA_SECRET", "ci-stub-secret")
os.environ.setdefault("ALPACA_SECRET_KEY", "ci-stub-secret")

import json
from pathlib import Path

import numpy as np
import pytest


def _seed_run(parent: Path, label: str, avg_pnl: float):
    d = parent / label
    d.mkdir(parents=True)
    payload = {
        "aggregate": {"avg_test_pnl": float(avg_pnl)},
        "folds": [{"per_episode_results": [
            {"symbol": "X", "date": "2024-12-01", "pnl": float(avg_pnl), "trades": 1}
        ]}],
    }
    (d / "quick_test_results.json").write_text(json.dumps(payload))


def test_discover_runs_groups_by_weight(tmp_path):
    from src.scripts.analyze_sweep import _discover_runs
    _seed_run(tmp_path, "w0.00_s1", -100.0)
    _seed_run(tmp_path, "w0.00_s2", -120.0)
    _seed_run(tmp_path, "w0.10_s1", -50.0)
    runs = _discover_runs(tmp_path)
    assert sorted(runs.keys()) == [0.0, 0.1]
    assert len(runs[0.0]) == 2
    assert len(runs[0.1]) == 1


def test_per_run_pnl_extracts_avg(tmp_path):
    from src.scripts.analyze_sweep import _per_run_pnl, _discover_runs
    _seed_run(tmp_path, "w0.20_s1", -75.5)
    runs = _discover_runs(tmp_path)
    pnl = _per_run_pnl(runs[0.2][0])
    assert pnl == -75.5


def test_bootstrap_mean_ci_brackets_mean():
    from src.scripts.analyze_sweep import _bootstrap_mean_ci
    vals = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5] * 5
    mean, lo, hi = _bootstrap_mean_ci(vals, n_iter=1000)
    assert lo < mean < hi


def test_bootstrap_mean_ci_returns_nan_when_too_few():
    from src.scripts.analyze_sweep import _bootstrap_mean_ci
    mean, lo, hi = _bootstrap_mean_ci([1.0, 2.0])
    assert np.isnan(lo) and np.isnan(hi)


def test_paired_bootstrap_delta_detects_improvement():
    from src.scripts.analyze_sweep import _paired_bootstrap_delta_ci
    a = [100.0, 110.0, 90.0, 105.0, 95.0]
    b = [0.0, 10.0, -10.0, 5.0, -5.0]
    delta, lo, hi = _paired_bootstrap_delta_ci(a, b, n_iter=1000)
    assert delta == pytest.approx(100.0, abs=1.0)
    assert lo > 0


def test_paired_bootstrap_delta_inconclusive_when_overlapping():
    from src.scripts.analyze_sweep import _paired_bootstrap_delta_ci
    a = [10.0, 11.0, 9.0, 10.5, 9.5]
    b = [10.0, 9.0, 11.0, 9.5, 10.5]
    _, lo, hi = _paired_bootstrap_delta_ci(a, b, n_iter=1000)
    assert lo < 0 < hi


def test_analyze_recommends_best_weight(tmp_path):
    for s, pnl in [(1, -150.0), (2, -140.0), (3, -160.0)]:
        _seed_run(tmp_path, f"w0.00_s{s}", pnl)
    for s, pnl in [(1, 50.0), (2, 60.0), (3, 40.0)]:
        _seed_run(tmp_path, f"w0.10_s{s}", pnl)
    for s, pnl in [(1, -155.0), (2, -145.0), (3, -150.0)]:
        _seed_run(tmp_path, f"w0.50_s{s}", pnl)
    from src.scripts.analyze_sweep import analyze
    report = analyze(tmp_path, baseline_weight=0.0)
    assert "Recommended weight: 0.10" in report
    assert "**yes**" in report


def test_analyze_handles_no_clear_winner(tmp_path):
    for s in range(1, 4):
        _seed_run(tmp_path, f"w0.00_s{s}", -100.0)
        _seed_run(tmp_path, f"w0.10_s{s}", -99.0)
    from src.scripts.analyze_sweep import analyze
    report = analyze(tmp_path, baseline_weight=0.0)
    assert "No clear winner" in report


def test_analyze_missing_baseline_skips_pairwise(tmp_path):
    for s in range(1, 4):
        _seed_run(tmp_path, f"w0.10_s{s}", -50.0)
    from src.scripts.analyze_sweep import analyze
    report = analyze(tmp_path, baseline_weight=0.0)
    assert "Per-weight summary" in report
    assert "baseline" in report.lower()
