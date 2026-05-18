"""Tests for diagnose_masks. Synthetic run.log fixtures."""
import os
os.environ.setdefault("ALPACA_API_KEY", "ci-stub-key")
os.environ.setdefault("ALPACA_SECRET", "ci-stub-secret")
os.environ.setdefault("ALPACA_SECRET_KEY", "ci-stub-secret")

from pathlib import Path


def _write_log(parent: Path, label: str, lines: list[str]) -> Path:
    d = parent / label
    d.mkdir(parents=True)
    p = d / "run.log"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_parse_counts_violations(tmp_path):
    from src.scripts.diagnose_masks import _parse_one_log
    log = _write_log(tmp_path, "w0.00_s1", [
        "INFO:__main__:Timesteps: 25000",
        "WARNING:src.rl.env:MASK VIOLATION #1: action_type=1 overridden to HOLD. mask=[0 0 1]",
        "INFO:parabolic_reversal:[TRAIN] Episode reset: X 2024-12-01",
        "WARNING:src.rl.env:MASK VIOLATION #2: action_type=0 overridden to HOLD. mask=[0 1 1]",
        "WARNING:src.rl.env:MASK VIOLATION #3: action_type=1 overridden to HOLD. mask=[0 1 1]",
        "INFO:parabolic_reversal:[TRAIN] Episode reset: Y 2024-12-02",
    ])
    s = _parse_one_log(log)
    assert s["total_violations"] == 3
    assert s["total_episodes"] == 2
    assert s["target_steps"] == 25000
    assert s["per_1000_steps"] == 3 / 25000 * 1000
    assert s["by_action_type"] == {0: 1, 1: 2}


def test_parse_detects_early_vs_late_drop(tmp_path):
    """A log where violations cluster in the first half should show drop."""
    from src.scripts.diagnose_masks import _parse_one_log
    lines = ["INFO:__main__:Timesteps: 25000"]
    # 4 episodes total: 2 early (many violations), 2 late (few)
    for ep in range(2):
        lines.append(f"INFO:parabolic_reversal:[TRAIN] Episode reset: SYM{ep} 2024-12-01")
        for _ in range(10):
            lines.append("WARNING:src.rl.env:MASK VIOLATION #1: action_type=1 overridden to HOLD")
    for ep in range(2, 4):
        lines.append(f"INFO:parabolic_reversal:[TRAIN] Episode reset: SYM{ep} 2024-12-01")
        lines.append("WARNING:src.rl.env:MASK VIOLATION #1: action_type=1 overridden to HOLD")
    log = _write_log(tmp_path, "w0.00_s1", lines)
    s = _parse_one_log(log)
    assert s["early_rate_per_episode"] > s["late_rate_per_episode"]


def test_analyze_smoke(tmp_path):
    from src.scripts.diagnose_masks import analyze
    _write_log(tmp_path, "w0.00_s1", [
        "INFO:__main__:Timesteps: 25000",
        "INFO:parabolic_reversal:[TRAIN] Episode reset",
        "WARNING:src.rl.env:MASK VIOLATION #1: action_type=1 overridden to HOLD",
        "INFO:parabolic_reversal:[TRAIN] Episode reset",
        "INFO:parabolic_reversal:[TRAIN] Episode reset",
        "INFO:parabolic_reversal:[TRAIN] Episode reset",
    ])
    report = analyze(tmp_path)
    assert "Per-run violation rates" in report
    assert "Per-weight aggregate" in report
    assert "0.00" in report


def test_analyze_flags_not_learning(tmp_path):
    """No drop early->late should produce 'Mask not being learned' message."""
    from src.scripts.diagnose_masks import analyze
    lines = ["INFO:__main__:Timesteps: 25000"]
    for ep in range(4):
        lines.append(f"INFO:parabolic_reversal:[TRAIN] Episode reset: S{ep}")
        for _ in range(5):
            lines.append("WARNING:src.rl.env:MASK VIOLATION #1: action_type=1 overridden to HOLD")
    _write_log(tmp_path, "w0.00_s1", lines)
    _write_log(tmp_path, "w0.00_s2", lines)
    _write_log(tmp_path, "w0.00_s3", lines)
    report = analyze(tmp_path)
    assert "Mask not being learned" in report


def test_analyze_handles_missing_logs(tmp_path):
    from src.scripts.diagnose_masks import analyze
    out = analyze(tmp_path)
    assert "No run.log files found" in out
