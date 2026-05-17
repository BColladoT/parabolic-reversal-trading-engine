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
    mean = float(arr.mean()) if arr.size else float("nan")
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
        d = float(a_arr.mean() - b_arr.mean()) if n > 0 else float("nan")
        return d, float("nan"), float("nan")
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
        lines.append(f"_(No runs at baseline weight {baseline_weight:.2f} - skipping pairwise.)_\n")
        return "\n".join(lines) + "\n"

    baseline_pnls = per_weight_pnls[baseline_weight]
    # Minimum economically-meaningful improvement (per-episode PnL, USD).
    # Combined with the CI-excludes-zero rule, this avoids declaring a winner
    # on trivially-small deltas that happen to be tight only because the
    # synthetic input had near-zero variance.
    min_effect = 5.0
    lines.append(f"## Pairwise vs baseline (weight={baseline_weight:.2f})\n")
    lines.append("| weight | delta | 95% paired-bootstrap CI of delta | beats baseline (95%) |")
    lines.append("|---|---:|---|:---:|")
    pairwise: list[tuple[float, float, float, float, bool]] = []
    for w, vals in per_weight_pnls.items():
        if w == baseline_weight:
            continue
        delta, lo, hi = _paired_bootstrap_delta_ci(vals, baseline_pnls)
        beats = (not np.isnan(lo)) and lo > 0 and delta >= min_effect
        ci = f"[{lo:+.2f}, {hi:+.2f}]" if not np.isnan(lo) else "n/a"
        lines.append(f"| {w:.2f} | {delta:+.2f} | {ci} | **{'yes' if beats else 'no'}** |")
        pairwise.append((w, delta, lo, hi, beats))

    lines.append("\n## Recommendation\n")
    winners = [t for t in pairwise if t[4]]
    if winners:
        winner = min(winners, key=lambda t: t[0])  # lowest weight that beats baseline
        lines.append(f"**Recommended weight: {winner[0]:.2f}**\n")
        lines.append(f"- Decision criterion: lowest weight whose paired-bootstrap CI for "
                     f"(weight - baseline) excludes zero")
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
