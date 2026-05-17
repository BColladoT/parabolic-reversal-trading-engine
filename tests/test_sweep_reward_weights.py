"""Tests for the reward-weight sweep orchestrator.

Mocks subprocess.run so no real training happens.
"""
import os
os.environ.setdefault("ALPACA_API_KEY", "ci-stub-key")
os.environ.setdefault("ALPACA_SECRET", "ci-stub-secret")
os.environ.setdefault("ALPACA_SECRET_KEY", "ci-stub-secret")

import json
from pathlib import Path
from unittest.mock import MagicMock, patch


def _fake_run_success_factory():
    """Returns a fake subprocess.run that writes a stub results.json and exits 0."""
    def fake_run(cmd, **kwargs):
        out_idx = cmd.index("--output-dir") + 1
        out_dir = Path(cmd[out_idx])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "quick_test_results.json").write_text(
            '{"aggregate": {"avg_test_pnl": -100.0}, "folds": []}'
        )
        return MagicMock(returncode=0, stdout=b"", stderr=b"")
    return fake_run


def test_sweep_runs_one_subprocess_per_weight_seed_pair(tmp_path):
    from src.scripts.sweep_reward_weights import run_sweep
    out_root = tmp_path / "sweep1"
    captured = []
    def fake_run(cmd, **kwargs):
        captured.append(list(cmd))
        return _fake_run_success_factory()(cmd, **kwargs)
    with patch("src.scripts.sweep_reward_weights.subprocess.run", side_effect=fake_run):
        run_sweep(weights=[0.0, 0.1], seeds=2, steps=5000, output_root=out_root)
    assert len(captured) == 4  # 2 weights * 2 seeds


def test_sweep_passes_correct_args_to_each_run(tmp_path):
    from src.scripts.sweep_reward_weights import run_sweep
    out_root = tmp_path / "sweep2"
    captured = []
    def fake_run(cmd, **kwargs):
        captured.append(list(cmd))
        return _fake_run_success_factory()(cmd, **kwargs)
    with patch("src.scripts.sweep_reward_weights.subprocess.run", side_effect=fake_run):
        run_sweep(weights=[0.1], seeds=1, steps=7000, output_root=out_root)
    cmd = captured[0]
    assert cmd[cmd.index("--r-multiple-reward-weight") + 1] == "0.1"
    assert cmd[cmd.index("--seed") + 1] == "1"
    assert cmd[cmd.index("--total-steps") + 1] == "7000"
    assert str(out_root) in cmd[cmd.index("--output-dir") + 1]


def test_sweep_skips_already_completed_runs(tmp_path):
    from src.scripts.sweep_reward_weights import run_sweep
    out_root = tmp_path / "sweep3"
    done = out_root / "w0.10_s1"
    done.mkdir(parents=True)
    (done / "quick_test_results.json").write_text("{}")
    captured = []
    def fake_run(cmd, **kwargs):
        captured.append(list(cmd))
        return _fake_run_success_factory()(cmd, **kwargs)
    with patch("src.scripts.sweep_reward_weights.subprocess.run", side_effect=fake_run):
        result = run_sweep(weights=[0.1], seeds=1, steps=5000, output_root=out_root)
    assert len(captured) == 0
    assert result["runs_completed"] == 0
    assert result["runs_skipped"] == 1


def test_sweep_writes_manifest(tmp_path):
    from src.scripts.sweep_reward_weights import run_sweep
    out_root = tmp_path / "sweep4"
    with patch("src.scripts.sweep_reward_weights.subprocess.run", side_effect=_fake_run_success_factory()):
        run_sweep(weights=[0.0, 0.2], seeds=2, steps=5000, output_root=out_root)
    manifest = json.loads((out_root / "sweep_manifest.json").read_text())
    assert manifest["weights"] == [0.0, 0.2]
    assert manifest["seeds"] == 2
    assert manifest["steps"] == 5000
    assert len(manifest["runs"]) == 4
    for r in manifest["runs"]:
        assert "weight" in r and "seed" in r and "output_dir" in r and "status" in r
        assert r["status"] in {"completed", "skipped", "failed"}


def test_sweep_handles_subprocess_failure_and_continues(tmp_path):
    from src.scripts.sweep_reward_weights import run_sweep
    out_root = tmp_path / "sweep5"
    call_count = {"n": 0}
    def fake_run(cmd, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return MagicMock(returncode=1, stdout=b"", stderr=b"simulated failure")
        return _fake_run_success_factory()(cmd, **kwargs)
    with patch("src.scripts.sweep_reward_weights.subprocess.run", side_effect=fake_run):
        result = run_sweep(weights=[0.0, 0.1], seeds=1, steps=5000, output_root=out_root)
    assert result["runs_completed"] == 1
    assert result["runs_failed"] == 1
    manifest = json.loads((out_root / "sweep_manifest.json").read_text())
    statuses = sorted(r["status"] for r in manifest["runs"])
    assert statuses == ["completed", "failed"]


def test_cli_main_parses_and_runs(tmp_path, monkeypatch, capsys):
    from src.scripts.sweep_reward_weights import main
    out_root = str(tmp_path / "cli_sweep")
    monkeypatch.setattr("sys.argv", [
        "sweep_reward_weights",
        "--weights", "0.0", "0.5",
        "--seeds", "1",
        "--steps", "5000",
        "--output-root", out_root,
    ])
    with patch("src.scripts.sweep_reward_weights.subprocess.run", side_effect=_fake_run_success_factory()):
        rc = main()
    assert rc == 0
    captured = capsys.readouterr()
    assert "Sweep complete" in captured.out
    assert "runs_completed: 2" in captured.out
