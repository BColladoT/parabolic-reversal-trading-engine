"""Tests for diagnose_training. Synthetic training_metrics.jsonl fixtures."""
import os
os.environ.setdefault("ALPACA_API_KEY", "ci-stub-key")
os.environ.setdefault("ALPACA_SECRET", "ci-stub-secret")
os.environ.setdefault("ALPACA_SECRET_KEY", "ci-stub-secret")

import json
from pathlib import Path


def _write_metrics(parent: Path, label: str, rewards: list[float]) -> None:
    d = parent / label
    d.mkdir(parents=True)
    with (d / "training_metrics.jsonl").open("w", encoding="utf-8") as f:
        for i, r in enumerate(rewards, 1):
            f.write(json.dumps({
                "iteration": i, "fold": 1,
                "timesteps": i * 400, "target_timesteps": len(rewards) * 400,
                "episode_reward_mean": r, "episode_reward_max": r,
                "episode_reward_min": r, "phase": "joint_training",
                "actor_lr": 3e-4, "scheduled_alpha": 0.1,
            }) + "\n")


def test_load_metrics_groups_by_weight_seed(tmp_path):
    from src.scripts.diagnose_training import _load_metrics
    _write_metrics(tmp_path, "w0.00_s1", [-10.0, -5.0, 0.0, 5.0])
    _write_metrics(tmp_path, "w0.00_s2", [-8.0, -3.0, 2.0])
    _write_metrics(tmp_path, "w0.10_s1", [1.0, 2.0])
    out = _load_metrics(tmp_path)
    assert sorted(out.keys()) == [0.0, 0.1]
    assert sorted(out[0.0].keys()) == [1, 2]
    assert len(out[0.0][1]) == 4


def test_summarize_finds_first_positive(tmp_path):
    from src.scripts.diagnose_training import _summarize_one_run, _load_metrics
    _write_metrics(tmp_path, "w0.00_s1", [-10.0, -5.0, -1.0, 0.5, 3.0])
    records = _load_metrics(tmp_path)[0.0][1]
    s = _summarize_one_run(records)
    assert s["first_positive_iter"] == 4
    assert s["final_reward"] == 3.0
    assert s["best_reward"] == 3.0
    assert s["n_iters"] == 5


def test_summarize_never_positive_returns_none(tmp_path):
    from src.scripts.diagnose_training import _summarize_one_run, _load_metrics
    _write_metrics(tmp_path, "w0.00_s1", [-10.0, -5.0, -1.0])
    records = _load_metrics(tmp_path)[0.0][1]
    s = _summarize_one_run(records)
    assert s["first_positive_iter"] is None


def test_analyze_flags_still_improving(tmp_path):
    """Late-quartile slope strongly positive -> 'still improving' message."""
    from src.scripts.diagnose_training import analyze
    # 20 iterations, monotonically improving last 5
    rewards = list(range(-20, 0))
    _write_metrics(tmp_path, "w0.00_s1", rewards)
    _write_metrics(tmp_path, "w0.00_s2", rewards)
    _write_metrics(tmp_path, "w0.00_s3", rewards)
    report = analyze(tmp_path)
    assert "Still improving" in report


def test_analyze_flags_plateau(tmp_path):
    """Flat late-quartile -> 'plateaued' message."""
    from src.scripts.diagnose_training import analyze
    # 20 iters; last 5 (late quartile) are flat at -1.0 -> slope ~0 -> plateau
    rewards = [-15.0] * 15 + [-1.0, -1.0, -1.0, -1.0, -1.0]
    _write_metrics(tmp_path, "w0.00_s1", rewards)
    _write_metrics(tmp_path, "w0.00_s2", rewards)
    _write_metrics(tmp_path, "w0.00_s3", rewards)
    report = analyze(tmp_path)
    assert "Plateaued" in report


def test_analyze_flags_high_variance(tmp_path):
    """Wildly different finals per seed -> 'high across-seed variance' message."""
    from src.scripts.diagnose_training import analyze
    _write_metrics(tmp_path, "w0.00_s1", [-15.0] * 18 + [-30.0, -50.0])
    _write_metrics(tmp_path, "w0.00_s2", [-15.0] * 18 + [+15.0, +30.0])
    _write_metrics(tmp_path, "w0.00_s3", [-15.0] * 18 + [0.0, +5.0])
    report = analyze(tmp_path)
    assert "across-seed variance" in report.lower()
