# Reward-Weight Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a professional, reproducible, statistically-sound sweep over `r_multiple_reward_weight ∈ {0.0, 0.1, 0.2, 0.5}` with 3 seeds each, plus an analyzer that ranks weights by paired-bootstrap CI and recommends the best.

**Architecture:** Two CLIs — `sweep_reward_weights.py` (subprocess-based orchestrator with manifest + resume) and `analyze_sweep.py` (pure-Python statistical analyzer with paired-bootstrap CIs and markdown output). Sequential GPU execution (one job at a time — 8GB VRAM). Resumability via per-run output-dir checks.

**Tech Stack:** Python 3.10, numpy (bootstrap), stdlib `subprocess` / `argparse` / `json`, existing `train_wfo_quick_test.py` from PR #8.

---

## Context

The first end-to-end RL training run (PR #8, merged `ae66425`) used `--r-multiple-reward-weight 0.5` and produced "High shaping ratio" warnings indicating the shaped term dominated the base Sortino reward by 6–13× on most steps. Operator intuition is that 0.1–0.2 is the right range. This plan codifies the verification.

**What we want from a "professional" sweep:**
- Multiple weights tested on the *same* fold/data (only `--r-multiple-reward-weight` varies)
- Multiple seeds per weight (so within-weight variance is estimable)
- Statistical comparison with paired-bootstrap CIs (since the same seed pairs across weights, comparisons can be paired for tighter intervals)
- Reproducible: every run logged, every config saved, every result archived
- Resumable: if the box reboots mid-sweep, we don't lose completed runs
- Bounded resource: sequential on the 8GB 3070
- Clear, opinionated decision criterion (no eyeballing)
- Unit-tested orchestrator + analyzer (smoke test before committing to a multi-hour real run)

**What we already have (per code inspection on `main`):**
- `train_wfo_quick_test.py` accepts `--r-multiple-reward-weight FLOAT` (PR #7) and `--output-dir PATH` (line 900) and `--total-steps INT` (line 902) and many other SAC hyperparams.
- `args.output_dir` flows into `QuickTestConfig` (line 950) → each run can write to a unique dir.
- Each run produces `<output-dir>/quick_test_results.json` with `aggregate.avg_test_pnl` (scalar) and `folds[0].per_episode_results` (list of `{symbol, date, pnl, trades}`).
- venv_ray310 is operational; GPU works; one quick test = ~12 min on RTX 3070.

**What we don't have:**
- `--seed` flag (without one, all runs at the same weight produce identical results — defeats the point of multiple seeds).
- A sweep orchestrator.
- A statistically-rigorous analyzer.

---

## File Structure

| File | Role |
|---|---|
| `src/scripts/train_wfo_quick_test.py` | MODIFY — add `--seed` CLI flag + propagation to torch / numpy / random / cuda |
| `src/scripts/sweep_reward_weights.py` | NEW — orchestrator (subprocess loop + manifest + resume) |
| `src/scripts/analyze_sweep.py` | NEW — analyzer (per-weight bootstrap CIs + paired delta vs baseline + recommendation) |
| `tests/test_seed_plumbing.py` | NEW — source-level + integration tests that `--seed` is wired correctly |
| `tests/test_sweep_reward_weights.py` | NEW — mock-subprocess tests for orchestrator |
| `tests/test_analyze_sweep.py` | NEW — synthetic-JSON tests for analyzer |

**Task dependency chain:** Task 1 → Tasks 2 & 3 (parallel, both depend on Task 1's `--seed` flag for the smoke test in Task 4 but their unit tests don't) → Task 4 (smoke; depends on 1+2+3) → Task 5 (real sweep; operator).

---

## Task 1: Add `--seed` flag to `train_wfo_quick_test.py`

**Files:**
- Modify: `src/scripts/train_wfo_quick_test.py` — add `--seed` flag and seed propagation
- Create: `tests/test_seed_plumbing.py`

- [ ] **Step 1: Read** `src/scripts/train_wfo_quick_test.py:895-980` to see the argparse block (line ~900) and the `QuickTestConfig` instantiation (line ~949). Confirm no `--seed` flag exists yet.

- [ ] **Step 2: Write failing tests** in `tests/test_seed_plumbing.py`:

```python
"""Verify --seed flag is wired into train_wfo_quick_test.py."""
import os
os.environ.setdefault("ALPACA_API_KEY", "ci-stub-key")
os.environ.setdefault("ALPACA_SECRET", "ci-stub-secret")
os.environ.setdefault("ALPACA_SECRET_KEY", "ci-stub-secret")

import re
from pathlib import Path


def test_quick_test_has_seed_flag_in_argparse():
    """Source-level: --seed flag is declared in the argparse block."""
    src = Path("src/scripts/train_wfo_quick_test.py").read_text()
    assert "'--seed'" in src or '"--seed"' in src, "missing --seed argparse declaration"


def test_quick_test_references_args_seed():
    """Source-level: args.seed is read somewhere after parsing."""
    src = Path("src/scripts/train_wfo_quick_test.py").read_text()
    assert "args.seed" in src, "args.seed never read after argparse"


def test_quick_test_propagates_seed_to_torch():
    """Source-level: torch.manual_seed(...) is called with the seed."""
    src = Path("src/scripts/train_wfo_quick_test.py").read_text()
    assert re.search(r"torch\.manual_seed\s*\(", src), "torch.manual_seed not called"


def test_quick_test_propagates_seed_to_numpy():
    """Source-level: numpy seeding via either np.random.seed or default_rng."""
    src = Path("src/scripts/train_wfo_quick_test.py").read_text()
    assert re.search(r"np\.random\.seed\s*\(|np\.random\.default_rng\s*\(", src), \
        "numpy seeding not present"


def test_quick_test_propagates_seed_to_python_random():
    """Source-level: stdlib random.seed call."""
    src = Path("src/scripts/train_wfo_quick_test.py").read_text()
    assert re.search(r"random\.seed\s*\(", src), "stdlib random.seed not called"
```

- [ ] **Step 3: Run, verify FAIL.**

Run: `pytest tests/test_seed_plumbing.py -v`
Expected: 5 fails (no `--seed` flag and no seed propagation).

- [ ] **Step 4: Implement.** In `src/scripts/train_wfo_quick_test.py`, find the argparse block near line 900 and add:

```python
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (torch + numpy + stdlib random + cuda) for reproducibility.')
```

In the same `main()` function, AFTER `args = parser.parse_args()` and BEFORE any model/env construction, add seed propagation:

```python
    # Reproducibility: seed every RNG that matters.
    import random as _stdlib_random
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    _stdlib_random.seed(args.seed)
```

`torch` and `np` are already imported at the top of the file (lines 13 and 20 respectively after PR #8 reordering).

- [ ] **Step 5: Run tests, verify PASS.**

Run: `pytest tests/test_seed_plumbing.py -v`
Expected: 5 pass.

- [ ] **Step 6: Smoke** (CPU import; should not actually train):

Run: `venv_ray310\Scripts\python.exe src\scripts\train_wfo_quick_test.py --help | grep -- --seed`
Expected: shows `--seed` line with the help text.

- [ ] **Step 7: Commit:**

```bash
git add src/scripts/train_wfo_quick_test.py tests/test_seed_plumbing.py
git commit -m "feat(rl): add --seed flag to train_wfo_quick_test for sweep reproducibility"
```

---

## Task 2: Build sweep orchestrator

**Files:**
- Create: `src/scripts/sweep_reward_weights.py`
- Create: `tests/test_sweep_reward_weights.py`

- [ ] **Step 1: Write failing tests** in `tests/test_sweep_reward_weights.py`:

```python
"""Tests for the reward-weight sweep orchestrator.

Mocks subprocess.run so no real training happens. The fake_run helper
mimics a successful train by creating quick_test_results.json in the
output dir the orchestrator picks.
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
        (out_dir / "quick_test_results.json").write_text('{"aggregate": {"avg_test_pnl": -100.0}, "folds": []}')
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
    # 2 weights x 2 seeds = 4 runs
    assert len(captured) == 4


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
    # Required flags are present
    assert "--r-multiple-reward-weight" in cmd
    assert cmd[cmd.index("--r-multiple-reward-weight") + 1] == "0.1"
    assert "--seed" in cmd
    assert cmd[cmd.index("--seed") + 1] == "1"
    assert "--total-steps" in cmd
    assert cmd[cmd.index("--total-steps") + 1] == "7000"
    assert "--output-dir" in cmd
    # Output dir matches the orchestrator's convention
    assert str(out_root) in cmd[cmd.index("--output-dir") + 1]


def test_sweep_skips_already_completed_runs(tmp_path):
    from src.scripts.sweep_reward_weights import run_sweep
    out_root = tmp_path / "sweep3"
    # Pre-create a completed run
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
    # Manifest reflects mixed statuses
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
```

- [ ] **Step 2: Run, verify FAIL.**

Run: `pytest tests/test_sweep_reward_weights.py -v`
Expected: 6 fails (module doesn't exist).

- [ ] **Step 3: Implement `src/scripts/sweep_reward_weights.py`:**

```python
"""Sweep r_multiple_reward_weight over a grid with multiple seeds.

Spawns one subprocess per (weight, seed) running train_wfo_quick_test.py
sequentially. Each run writes to its own output dir under
``<output_root>/wW.WW_sN/``. Resumable: if a run's quick_test_results.json
already exists, the run is skipped on the next invocation.

Usage:
    python -m src.scripts.sweep_reward_weights \\
        --weights 0.0 0.1 0.2 0.5 \\
        --seeds 3 \\
        --steps 25000 \\
        --output-root models/sweep_20260518/

After completion:
    python -m src.scripts.analyze_sweep models/sweep_20260518/
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List


def _run_dir(output_root: Path, weight: float, seed: int) -> Path:
    return output_root / f"w{weight:.2f}_s{seed}"


def _already_done(run_dir: Path) -> bool:
    return (run_dir / "quick_test_results.json").exists()


def _run_one(weight: float, seed: int, steps: int, output_root: Path,
             python_exe: str = sys.executable) -> dict:
    run_dir = _run_dir(output_root, weight, seed)
    if _already_done(run_dir):
        return {"weight": weight, "seed": seed,
                "output_dir": str(run_dir),
                "status": "skipped",
                "reason": "quick_test_results.json already exists"}

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"

    cmd = [
        python_exe,
        "src/scripts/train_wfo_quick_test.py",
        "--r-multiple-reward-weight", str(weight),
        "--seed", str(seed),
        "--total-steps", str(steps),
        "--output-dir", str(run_dir),
    ]

    print(f"[sweep] running weight={weight} seed={seed} -> {run_dir}", flush=True)
    with log_path.open("wb") as log_f:
        result = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        return {"weight": weight, "seed": seed,
                "output_dir": str(run_dir),
                "status": "failed",
                "returncode": result.returncode,
                "log": str(log_path)}

    if not _already_done(run_dir):
        return {"weight": weight, "seed": seed,
                "output_dir": str(run_dir),
                "status": "failed",
                "returncode": 0,
                "log": str(log_path),
                "error": "process exited 0 but no results.json"}

    return {"weight": weight, "seed": seed,
            "output_dir": str(run_dir),
            "status": "completed",
            "log": str(log_path)}


def run_sweep(weights: List[float], seeds: int, steps: int,
              output_root: Path, python_exe: str = sys.executable) -> dict:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    runs: list[dict] = []
    for weight in weights:
        for seed in range(1, seeds + 1):
            runs.append(_run_one(weight, seed, steps, output_root, python_exe))

    manifest = {
        "weights": list(weights),
        "seeds": seeds,
        "steps": steps,
        "output_root": str(output_root),
        "started_at": datetime.utcnow().isoformat() + "Z",
        "runs": runs,
    }
    (output_root / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2))

    return {
        "runs_completed": sum(1 for r in runs if r["status"] == "completed"),
        "runs_skipped":   sum(1 for r in runs if r["status"] == "skipped"),
        "runs_failed":    sum(1 for r in runs if r["status"] == "failed"),
        "manifest_path":  str(output_root / "sweep_manifest.json"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Sweep r_multiple_reward_weight grid.")
    p.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.5],
                   help="Weights to sweep (default: 0.0 0.1 0.2 0.5).")
    p.add_argument("--seeds", type=int, default=3,
                   help="Number of seeds per weight (default: 3).")
    p.add_argument("--steps", type=int, default=25000,
                   help="Training steps per run (default: 25000).")
    p.add_argument("--output-root", type=Path,
                   default=Path(f"models/sweep_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}/"),
                   help="Root output directory (default: timestamped under models/).")
    p.add_argument("--python", default=sys.executable,
                   help="Python interpreter for child processes (default: this interpreter).")
    args = p.parse_args()

    stats = run_sweep(args.weights, args.seeds, args.steps,
                      args.output_root, args.python)
    print("\n=== Sweep complete ===")
    print(f"  runs_completed: {stats['runs_completed']}")
    print(f"  runs_skipped:   {stats['runs_skipped']}")
    print(f"  runs_failed:    {stats['runs_failed']}")
    print(f"  manifest:       {stats['manifest_path']}")
    print("\nNext step:")
    print(f"  python -m src.scripts.analyze_sweep {args.output_root}")
    return 0 if stats["runs_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests, verify PASS.**

Run: `pytest tests/test_sweep_reward_weights.py -v`
Expected: 6 pass.

- [ ] **Step 5: Commit:**

```bash
git add src/scripts/sweep_reward_weights.py tests/test_sweep_reward_weights.py
git commit -m "feat(scripts): sweep_reward_weights orchestrator with manifest + resume"
```

---

## Task 3: Build sweep analyzer with paired-bootstrap CIs

**Files:**
- Create: `src/scripts/analyze_sweep.py`
- Create: `tests/test_analyze_sweep.py`

- [ ] **Step 1: Write failing tests** in `tests/test_analyze_sweep.py`:

```python
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
    """Drop a fixture quick_test_results.json into parent/label/."""
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
    """b is consistently ~100 worse than a; delta should be ~+100 with positive CI."""
    from src.scripts.analyze_sweep import _paired_bootstrap_delta_ci
    a = [100.0, 110.0, 90.0, 105.0, 95.0]
    b = [0.0, 10.0, -10.0, 5.0, -5.0]
    delta, lo, hi = _paired_bootstrap_delta_ci(a, b, n_iter=1000)
    assert delta == pytest.approx(100.0, abs=1.0)
    assert lo > 0  # confidently positive


def test_paired_bootstrap_delta_inconclusive_when_overlapping():
    """When a and b have overlapping distributions, CI should bracket 0."""
    from src.scripts.analyze_sweep import _paired_bootstrap_delta_ci
    a = [10.0, 11.0, 9.0, 10.5, 9.5]
    b = [10.0, 9.0, 11.0, 9.5, 10.5]
    _, lo, hi = _paired_bootstrap_delta_ci(a, b, n_iter=1000)
    assert lo < 0 < hi


def test_analyze_recommends_best_weight(tmp_path):
    """Builds a sweep where 0.10 clearly beats 0.00; analyzer should pick 0.10."""
    for s, pnl in [(1, -150.0), (2, -140.0), (3, -160.0)]:
        _seed_run(tmp_path, f"w0.00_s{s}", pnl)
    for s, pnl in [(1, 50.0), (2, 60.0), (3, 40.0)]:
        _seed_run(tmp_path, f"w0.10_s{s}", pnl)
    for s, pnl in [(1, -155.0), (2, -145.0), (3, -150.0)]:
        _seed_run(tmp_path, f"w0.50_s{s}", pnl)
    from src.scripts.analyze_sweep import analyze
    report = analyze(tmp_path, baseline_weight=0.0)
    assert "Recommended weight: 0.10" in report
    assert "**yes**" in report  # at least one weight beats baseline


def test_analyze_handles_no_clear_winner(tmp_path):
    """Baseline and candidate are statistically indistinguishable."""
    for s in range(1, 4):
        _seed_run(tmp_path, f"w0.00_s{s}", -100.0)
        _seed_run(tmp_path, f"w0.10_s{s}", -99.0)  # tiny diff
    from src.scripts.analyze_sweep import analyze
    report = analyze(tmp_path, baseline_weight=0.0)
    assert "No clear winner" in report


def test_analyze_missing_baseline_skips_pairwise(tmp_path):
    """If no runs at baseline_weight, analyzer still produces a summary."""
    for s in range(1, 4):
        _seed_run(tmp_path, f"w0.10_s{s}", -50.0)
    from src.scripts.analyze_sweep import analyze
    report = analyze(tmp_path, baseline_weight=0.0)
    assert "Per-weight summary" in report
    assert "baseline" in report.lower()  # mentions missing baseline somewhere
```

- [ ] **Step 2: Run, verify FAIL.**

Run: `pytest tests/test_analyze_sweep.py -v`
Expected: 9 fails (module doesn't exist).

- [ ] **Step 3: Implement `src/scripts/analyze_sweep.py`:**

```python
"""Analyze a reward-weight sweep produced by sweep_reward_weights.py.

Reads every quick_test_results.json under the sweep root, groups by weight,
computes per-weight bootstrap CIs, and pairwise paired-bootstrap CIs of
(weight - baseline) deltas. Picks the lowest weight whose delta CI excludes
zero against the baseline.

Usage:
    python -m src.scripts.analyze_sweep models/sweep_20260518/
    python -m src.scripts.analyze_sweep models/sweep_20260518/ --baseline-weight 0.0
    python -m src.scripts.analyze_sweep models/sweep_20260518/ --output reports/sweep.md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _discover_runs(sweep_root: Path) -> Dict[float, List[Path]]:
    """Group quick_test_results.json paths by weight (parsed from dirname wW.WW_sN)."""
    runs: Dict[float, List[Path]] = defaultdict(list)
    for results_path in Path(sweep_root).glob("w*_s*/quick_test_results.json"):
        try:
            weight_str = results_path.parent.name.split("_")[0][1:]  # 'w0.10' -> '0.10'
            weight = float(weight_str)
            runs[weight].append(results_path)
        except (ValueError, IndexError):
            continue
    return dict(runs)


def _per_run_pnl(results_path: Path) -> float:
    """Extract avg test PnL from a single quick_test_results.json."""
    d = json.loads(results_path.read_text())
    return float(d.get("aggregate", {}).get("avg_test_pnl", 0.0))


def _bootstrap_mean_ci(values, n_iter: int = 2000, alpha: float = 0.05,
                       seed: int = 42) -> Tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) via percentile bootstrap.

    Returns (mean, NaN, NaN) when len(values) < 3 (too few for any signal).
    """
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if arr.size < 3:
        return mean, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    resamples = rng.choice(arr, size=(n_iter, arr.size), replace=True)
    boot = resamples.mean(axis=1)
    return mean, float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


def _paired_bootstrap_delta_ci(a, b, n_iter: int = 2000, alpha: float = 0.05,
                               seed: int = 42) -> Tuple[float, float, float]:
    """Paired-bootstrap CI of (mean(a) - mean(b)).

    Pairs a[i] and b[i] (same seed index across weights). Resamples seed
    indices with replacement. Returns (delta, NaN, NaN) for n < 3.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    n = min(a_arr.size, b_arr.size)
    if n < 3:
        return float(a_arr.mean() - b_arr.mean()), float("nan"), float("nan")
    a_arr = a_arr[:n]
    b_arr = b_arr[:n]
    delta = float(a_arr.mean() - b_arr.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_iter, n))
    boot = a_arr[idx].mean(axis=1) - b_arr[idx].mean(axis=1)
    return delta, float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


def analyze(sweep_root: Path, baseline_weight: float = 0.0) -> str:
    """Produce a markdown report string."""
    runs_by_weight = _discover_runs(sweep_root)
    if not runs_by_weight:
        return f"No runs found under `{sweep_root}`.\n"

    per_weight_pnls: Dict[float, List[float]] = {
        w: [_per_run_pnl(p) for p in sorted(paths)]
        for w, paths in sorted(runs_by_weight.items())
    }

    lines: list[str] = []
    lines.append("# Reward-Weight Sweep Analysis\n")
    lines.append(f"**Sweep root:** `{sweep_root}`")
    lines.append(f"**Weights found:** {sorted(per_weight_pnls.keys())}")
    lines.append(f"**Seeds per weight (min/max):** "
                 f"{min(len(v) for v in per_weight_pnls.values())} / "
                 f"{max(len(v) for v in per_weight_pnls.values())}\n")

    lines.append("## Per-weight summary\n")
    lines.append("| weight | n_seeds | mean_pnl | 95% bootstrap CI |")
    lines.append("|---|---:|---:|---|")
    for w, vals in per_weight_pnls.items():
        mean, lo, hi = _bootstrap_mean_ci(vals)
        ci = f"[{lo:+.2f}, {hi:+.2f}]" if not np.isnan(lo) else "n/a (need >=3 seeds)"
        lines.append(f"| {w:.2f} | {len(vals)} | {mean:+.2f} | {ci} |")
    lines.append("")

    if baseline_weight not in per_weight_pnls:
        lines.append(f"## Pairwise vs baseline\n")
        lines.append(f"_(No runs at baseline weight {baseline_weight:.2f} — skipping pairwise.)_\n")
        return "\n".join(lines) + "\n"

    baseline_pnls = per_weight_pnls[baseline_weight]
    lines.append(f"## Pairwise vs baseline (weight={baseline_weight:.2f})\n")
    lines.append("| weight | delta | 95% paired-bootstrap CI of delta | beats baseline (95%) |")
    lines.append("|---|---:|---|:---:|")
    pairwise: list[tuple[float, float, float, float, bool]] = []
    for w, vals in per_weight_pnls.items():
        if w == baseline_weight:
            continue
        delta, lo, hi = _paired_bootstrap_delta_ci(vals, baseline_pnls)
        beats = (not np.isnan(lo)) and lo > 0
        ci = f"[{lo:+.2f}, {hi:+.2f}]" if not np.isnan(lo) else "n/a"
        lines.append(f"| {w:.2f} | {delta:+.2f} | {ci} | **{'yes' if beats else 'no'}** |")
        pairwise.append((w, delta, lo, hi, beats))

    lines.append("\n## Recommendation\n")
    winners = [t for t in pairwise if t[4]]
    if winners:
        winner = min(winners, key=lambda t: t[0])  # lowest weight that beats baseline
        lines.append(f"**Recommended weight: {winner[0]:.2f}**\n")
        lines.append(f"- Decision criterion: lowest weight whose paired-bootstrap CI for "
                     f"(weight − baseline) excludes zero")
        lines.append(f"- Lower-CI bound on improvement vs baseline: +{winner[2]:.2f} / episode (95%)")
        lines.append(f"- Point estimate of improvement: +{winner[1]:.2f} / episode")
    else:
        lines.append("**No clear winner.**")
        lines.append("\nNo weight statistically beats the baseline at the 95% level.")
        lines.append("Consider: more seeds for tighter CIs, or expand the weight grid.")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("sweep_root", type=Path)
    p.add_argument("--baseline-weight", type=float, default=0.0,
                   help="Weight to use as comparison baseline (default: 0.0).")
    p.add_argument("--output", type=Path, default=None,
                   help="Optional: write report to file instead of stdout.")
    args = p.parse_args()
    report = analyze(args.sweep_root, args.baseline_weight)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        print(f"Report written to: {args.output}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests, verify PASS.**

Run: `pytest tests/test_analyze_sweep.py -v`
Expected: 9 pass.

- [ ] **Step 5: Commit:**

```bash
git add src/scripts/analyze_sweep.py tests/test_analyze_sweep.py
git commit -m "feat(scripts): analyze_sweep with paired-bootstrap CIs + winner selection"
```

---

## Task 4: Smoke test the full sweep+analyze pipeline

**Operator action — verifies code before committing to a multi-hour real run.**

- [ ] **Step 1: Confirm prior tasks are merged or at least local.** Run:

```
git log --oneline -3
git status --short
```

Expected: clean tree on `main` (or feature branch) with the 3 commits from Tasks 1–3 present.

- [ ] **Step 2: Run the analyzer test suite once more, end-to-end:**

```
python -m pytest tests/test_seed_plumbing.py tests/test_sweep_reward_weights.py tests/test_analyze_sweep.py -v
```

Expected: all green (20 tests).

- [ ] **Step 3: Run a 2-job smoke sweep** (1 seed × 2 weights × 5K steps = ~10 min total):

```
venv_ray310\Scripts\python.exe -m src.scripts.sweep_reward_weights ^
    --weights 0.0 0.5 ^
    --seeds 1 ^
    --steps 5000 ^
    --output-root models/sweep_smoke/
```

Expected at completion:
- `models/sweep_smoke/w0.00_s1/quick_test_results.json` exists
- `models/sweep_smoke/w0.50_s1/quick_test_results.json` exists
- `models/sweep_smoke/sweep_manifest.json` exists
- Stdout ends with: `runs_completed: 2 / runs_skipped: 0 / runs_failed: 0`

- [ ] **Step 4: Run the analyzer on the smoke results:**

```
python -m src.scripts.analyze_sweep models/sweep_smoke/
```

Expected output: markdown report with per-weight summary (CIs will print `n/a (need >=3 seeds)`) and pairwise table for 0.50 vs 0.00 (delta CI also n/a for n=1). Recommendation section will say "No clear winner" since n<3 — that's fine; we're verifying plumbing.

- [ ] **Step 5: Verify resumability.** Re-run the smoke command from Step 3 verbatim. Expected:
- Stdout ends with: `runs_completed: 0 / runs_skipped: 2 / runs_failed: 0`
- No new training runs are launched (existing results.json files are detected).

- [ ] **Step 6: If all of Steps 3–5 succeeded, proceed to Task 5.** If anything failed, fix and re-smoke; do NOT proceed to a multi-hour run that will fail the same way.

---

## Task 5: Run the real sweep + generate report (operator action)

**Long-running operator action. Expected wall time: 2.5–3 hours on RTX 3070.**

- [ ] **Step 1: Pre-flight checks.**

```
nvidia-smi
python -m src.scripts.preflight_paper_trade
```

Confirm GPU is idle (no other training running) and temperature is reasonable (< 75 °C is ideal — the laptop got to 83 °C idle in earlier runs). The preflight is unrelated to training but a good general health check.

- [ ] **Step 2: Kick off the real sweep** (12 runs total: 4 weights × 3 seeds × 25 K steps):

```
venv_ray310\Scripts\python.exe -m src.scripts.sweep_reward_weights ^
    --weights 0.0 0.1 0.2 0.5 ^
    --seeds 3 ^
    --steps 25000 ^
    --output-root models/sweep_2026-05-18/
```

If you must stop and resume: Ctrl-C the orchestrator; on restart, completed runs are skipped automatically (verified in Task 4 Step 5).

- [ ] **Step 3: While it runs, optionally monitor:**

```
tasklist | findstr python
nvidia-smi -l 30
type models\sweep_2026-05-18\w0.10_s1\run.log
```

- [ ] **Step 4: After completion, run the analyzer and save the report:**

```
python -m src.scripts.analyze_sweep models\sweep_2026-05-18\ ^
    --output reports\sweep_2026-05-18.md
type reports\sweep_2026-05-18.md
```

Expected: markdown report with a clear `Recommended weight: X.XX` line (or `No clear winner` if the candidates are all statistically indistinguishable from baseline — in which case, expand the grid).

- [ ] **Step 5: Open a PR with the report** (optional, recommended for record-keeping):

```
git checkout -b chore/sweep-results-2026-05-18
git add reports/sweep_2026-05-18.md
git commit -m "chore(rl): record reward-weight sweep results 2026-05-18"
git push -u origin chore/sweep-results-2026-05-18
gh pr create --title "chore: reward-weight sweep results 2026-05-18" ^
    --body "$(type reports\sweep_2026-05-18.md)"
gh pr merge --auto --squash
```

- [ ] **Step 6 (optional): Update the documented default reward weight.** If the sweep clearly recommends a specific weight, edit the help text in both training scripts to mention "recommended: X.XX". Separate commit on the same PR. Do NOT change the default value silently — operators may have scripts pinning the current default.

---

## Out of Scope (explicitly NOT in this plan)

- **Full-WFO sweep** (multiple folds per run): production WFO runs are 6–12 hours each; sweeping 12 of them is days of compute. Use the quick-test signal first; only fold-validate the winner in a follow-up.
- **Bayesian / Optuna hyperparameter search:** overkill for a 4-point grid. Re-evaluate if we ever sweep multiple hyperparameters jointly.
- **CI gate on sweep results:** the sweep takes hours; can't run in GitHub Actions. Tests cover only orchestrator + analyzer correctness against mocks/fixtures.
- **Matplotlib plots:** markdown report is sufficient. Add plotting if visual inspection becomes load-bearing.
- **Slack notification on sweep completion:** trivial to add later (`send_alert(...)` after the manifest write), not in scope here.
- **Sweep on other hyperparameters** (alpha schedule, batch size, etc.): out of scope. This plan is specifically about `r_multiple_reward_weight`.

---

## Why this design is "professional"

| Property | How it's achieved |
|---|---|
| Reproducible | Explicit `--seed` flag plumbed to torch + cuda + numpy + stdlib random; sweep manifest logs all configs and timestamps |
| Statistically rigorous | 3 seeds per weight (variance estimation); 95% bootstrap CIs per weight; paired-bootstrap for (weight − baseline) deltas |
| Resumable | `_already_done()` check via results.json existence; killed sweeps restart cleanly with completed runs skipped |
| Comparable | Identical fold, step budget, and SAC hyperparams across all runs; only `--r-multiple-reward-weight` and `--seed` differ |
| Auditable | Per-run `run.log` capturing stdout+stderr, per-run `quick_test_results.json`, sweep `sweep_manifest.json`, markdown report committed to `reports/` |
| Resource-bounded | Sequential subprocess execution (8 GB VRAM constraint); no parallel job risk of OOM |
| Opinionated decision | "Lowest weight whose paired-bootstrap CI for (weight − baseline) excludes zero" — codified in the analyzer, no eyeballing |
| Tested | Orchestrator + analyzer have unit tests with mocked subprocess + synthetic JSON fixtures; smoke test on tiny grid before committing to the multi-hour real run |
