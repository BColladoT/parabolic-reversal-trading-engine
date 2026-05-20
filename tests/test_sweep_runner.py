"""
Smoke test for src/scripts/run_sweep.py.

Uses --dry-run mode (no real trainer subprocess) so this test runs in
seconds. The real trainer integration is exercised separately by Phase 3
GPU sweeps. This test verifies the sweep runner's plumbing:
  * Cartesian product over (sweep_values x seeds)
  * Per-config aggregation (mean / median / std)
  * Summary JSON shape matches consumers' expectations
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile


def test_sweep_runner_produces_aggregated_json():
    """End-to-end smoke test of run_sweep.py in --dry-run mode."""
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "sweep_out"
        # Use sys.executable to avoid PATH/venv mismatches between Windows
        # shells and the harness; on Windows venv_ray310 is what the user runs
        # interactively, but invoking sys.executable picks up whatever
        # interpreter is running pytest (which is the same venv).
        result = subprocess.run(
            [
                sys.executable,
                "src/scripts/run_sweep.py",
                "--algo", "ppo",
                "--action-space", "discrete",
                "--total-steps", "1000",
                "--sweep", "discrete_action_bins=3,5",
                "--n-seeds", "2",
                "--output-dir", str(out),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0, (
            f"run_sweep.py failed (rc={result.returncode}).\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

        summary_path = out / "sweep_summary.json"
        assert summary_path.exists(), f"sweep_summary.json missing at {summary_path}"
        summary = json.loads(summary_path.read_text())

        assert summary["param"] == "discrete_action_bins"
        assert summary["values"] == ["3", "5"]
        assert len(summary["configs"]) == 2  # 3-bin, 5-bin

        for c in summary["configs"]:
            assert c["param"] == "discrete_action_bins"
            assert len(c["per_seed_results"]) == 2  # 2 seeds
            assert "mean_test_pnl" in c
            assert "median_test_pnl" in c
            assert "std_test_pnl" in c
            assert "n_failed" in c
            # Dry-run results should all succeed
            assert c["n_failed"] == 0
            assert c["mean_test_pnl"] is not None
            assert c["std_test_pnl"] is not None
            # Each per-seed result must carry a seed and a test_pnl
            for r in c["per_seed_results"]:
                assert "seed" in r
                assert "test_pnl" in r
                assert r["test_pnl"] is not None
