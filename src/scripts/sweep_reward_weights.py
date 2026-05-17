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
