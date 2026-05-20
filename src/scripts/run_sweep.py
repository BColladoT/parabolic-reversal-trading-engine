"""Sweep runner: cartesian product of (sweep_axis x seeds) -> results table.

Phase 2 of the RL tuning plan. Phases 3-6 will USE this runner to drive
training budget sweeps, bin-count sweeps, PPO HP sweeps, and the final
10-seed confirmation. Without it, each sweep is a manual N-config x M-seed
loop of CLI invocations.

Usage:
    python src/scripts/run_sweep.py \\
        --algo ppo --action-space discrete --total-steps 25000 \\
        --sweep discrete_action_bins=3,5,7,9,11 \\
        --n-seeds 3 \\
        --output-dir models/sweep_bins_2026-05-20

Each (value, seed) pair runs ``src/scripts/train_wfo_quick_test.py`` in a
subprocess; outputs are written to
``<output-dir>/<param>=<value>_seed=<seed>/`` and aggregated into
``<output-dir>/sweep_summary.json`` for downstream analysis (Phase 6).

The summary JSON keeps the raw per-seed payloads so consumers can apply
``src/utils/statistical_tests.py`` (bootstrap CI, permutation test) without
losing information. This script intentionally does NOT recreate those
helpers (YAGNI; see Phase 2 plan note re ``sweep_aggregator.py``).

``--dry-run`` short-circuits the subprocess and emits synthetic
``quick_test_results.json`` payloads with a deterministic
seed-derived ``test_pnl_total``. Used by ``tests/test_sweep_runner.py``
and useful for sanity-checking the plumbing on a laptop.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


def _parse_sweep(spec: str) -> tuple[str, list[str]]:
    """Parse ``key=v1,v2,v3`` into ``(key, [v1, v2, v3])``.

    Splits on the FIRST '=' only (values may contain further '=' for
    nested specs, though we don't use that today). Empty values are
    rejected loudly because they almost always indicate a typo.
    """
    if "=" not in spec:
        raise ValueError(
            f"--sweep must be 'key=v1,v2,...'; got {spec!r} (no '=')"
        )
    key, raw_vals = spec.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"--sweep key is empty in {spec!r}")
    values = [v.strip() for v in raw_vals.split(",")]
    if any(not v for v in values):
        raise ValueError(
            f"--sweep has empty value in {spec!r}; check for trailing or "
            f"adjacent commas"
        )
    return key, values


def _synthetic_payload(seed: int, value: str) -> dict:
    """Deterministic stand-in for ``quick_test_results.json``.

    Used by ``--dry-run`` so the test suite (and humans) can exercise
    the sweep plumbing without burning ~10 min on a real training run.
    The exact PnL formula doesn't matter as long as it is deterministic
    and produces non-trivial variation across (seed, value).
    """
    # Stable hash-ish derived from value so different sweep values get
    # different baselines without colliding on seed.
    try:
        value_offset = float(value)
    except ValueError:
        value_offset = float(sum(ord(c) for c in value) % 100)
    test_pnl = -1000.0 + seed * 100.0 + value_offset * 10.0
    return {
        "fold_metrics": {
            "test_pnl_total": test_pnl,
            "win_rate": 0.5 + (seed % 3) * 0.05,
            "mean_episode_pnl": test_pnl / 50.0,
        },
        "action_distribution": None,
        "config": {"dry_run": True, "seed": seed, "value": value},
    }


def _run_one(
    common_args: list[str],
    param: str,
    value: str,
    seed: int,
    run_out: Path,
    *,
    dry_run: bool = False,
) -> dict:
    """Run a single trainer invocation and harvest its results JSON.

    Returns a dict that always contains ``seed`` and ``wall_time_s``.
    On success: ``test_pnl``, ``win_rate``, ``action_distribution``.
    On failure: ``error`` (tail of stderr) instead of ``test_pnl``.

    Bridges the quick-test vs full-WFO JSON-key mismatch:
    quick-test writes ``test_pnl_total`` (under ``fold_metrics`` or at
    top level), full-WFO writes ``total_test_pnl``.
    """
    results_json = run_out / "quick_test_results.json"
    t0 = time.time()

    if dry_run:
        payload = _synthetic_payload(seed, value)
        run_out.mkdir(parents=True, exist_ok=True)
        results_json.write_text(json.dumps(payload, indent=2))
        res_stderr = ""
        rc = 0
    else:
        flag = "--" + param.replace("_", "-")
        cmd = [
            sys.executable,
            "src/scripts/train_wfo_quick_test.py",
            *common_args,
            flag,
            value,
            "--seed",
            str(seed),
            "--output-dir",
            str(run_out),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        rc = res.returncode
        res_stderr = res.stderr or ""

    wall = time.time() - t0

    if rc != 0 or not results_json.exists():
        return {
            "seed": seed,
            "error": res_stderr[-2000:],
            "wall_time_s": wall,
        }

    payload = json.loads(results_json.read_text())
    # Quick-test writes test_pnl_total (nested under fold_metrics);
    # full-WFO writes total_test_pnl. Some older runs put it at top level.
    fold = payload.get("fold_metrics", payload)
    test_pnl: Optional[float] = fold.get("test_pnl_total")
    if test_pnl is None:
        test_pnl = fold.get("total_test_pnl")
    return {
        "seed": seed,
        "test_pnl": test_pnl,
        "win_rate": fold.get("win_rate"),
        "action_distribution": payload.get("action_distribution"),
        "wall_time_s": wall,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run (config x seed) sweep of train_wfo_quick_test.py.",
    )
    ap.add_argument("--algo", required=True, choices=["sac", "ppo"])
    ap.add_argument(
        "--action-space", required=True, choices=["continuous", "discrete"]
    )
    ap.add_argument("--total-steps", required=True)
    ap.add_argument(
        "--sweep",
        required=True,
        help="Sweep spec: 'param_name=v1,v2,v3'. The param is forwarded to "
        "the trainer as --param-name (underscores -> dashes).",
    )
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help="Extra flags forwarded verbatim to the trainer (e.g. "
        "--extra --lr-actor 3e-4 --hold-band-threshold 0.3).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the real trainer subprocess; emit deterministic "
        "synthetic quick_test_results.json payloads. Used by tests.",
    )
    args = ap.parse_args()

    param, values = _parse_sweep(args.sweep)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.base_seed, args.base_seed + args.n_seeds))
    common = [
        "--algo", args.algo,
        "--action-space", args.action_space,
        "--total-steps", str(args.total_steps),
        *args.extra,
    ]

    configs = []
    for value in values:
        per_seed: list[dict] = []
        for seed in seeds:
            run_out = out_root / f"{param}={value}_seed={seed}"
            run_out.mkdir(parents=True, exist_ok=True)
            r = _run_one(common, param, value, seed, run_out, dry_run=args.dry_run)
            per_seed.append(r)
            pnl_repr = r.get("test_pnl")
            pnl_str = f"{pnl_repr:.2f}" if isinstance(pnl_repr, (int, float)) else "ERR"
            print(
                f"[{param}={value} seed={seed}] test_pnl={pnl_str} "
                f"t={r['wall_time_s']:.0f}s"
            )

        pnls = [
            r["test_pnl"]
            for r in per_seed
            if "test_pnl" in r and r["test_pnl"] is not None
        ]
        configs.append({
            "param": param,
            "value": value,
            "per_seed_results": per_seed,
            "mean_test_pnl": float(np.mean(pnls)) if pnls else None,
            "median_test_pnl": float(np.median(pnls)) if pnls else None,
            # ddof=1 matches sample stddev used elsewhere in the codebase
            # (Phase 0 fix-up). Needs >=2 samples to be defined.
            "std_test_pnl": float(np.std(pnls, ddof=1)) if len(pnls) >= 2 else None,
            "n_failed": sum(1 for r in per_seed if "error" in r),
        })

    summary = {
        "configs": configs,
        "param": param,
        "values": values,
        "seeds": seeds,
        "algo": args.algo,
        "action_space": args.action_space,
        "total_steps": args.total_steps,
        "dry_run": args.dry_run,
    }
    (out_root / "sweep_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n--- SUMMARY ---")
    for c in configs:
        mean_str = (
            f"${c['mean_test_pnl']:.0f}" if c["mean_test_pnl"] is not None else "ERR"
        )
        med_str = (
            f"${c['median_test_pnl']:.0f}"
            if c["median_test_pnl"] is not None
            else "ERR"
        )
        std_str = (
            f"${c['std_test_pnl']:.0f}" if c["std_test_pnl"] is not None else "n/a"
        )
        print(
            f"{c['param']}={c['value']:<8}  mean={mean_str:>10}  "
            f"median={med_str:>10}  std={std_str:>8}  (failed: {c['n_failed']})"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
