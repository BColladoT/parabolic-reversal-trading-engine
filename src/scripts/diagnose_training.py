"""Training-curve diagnostic on a completed reward-weight sweep.

Reads training_metrics.jsonl files, computes convergence indicators per
(weight, seed): final reward, best reward, time-to-first-positive,
late-iteration variance (proxy for "did it converge or was it still
bouncing"), and improvement-trend over last quartile.

Usage:
    python -m src.scripts.diagnose_training models/sweep_2026-05-18/
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def _load_metrics(sweep_root: Path) -> Dict[float, Dict[int, List[dict]]]:
    """Returns weight -> seed -> list of iteration records."""
    out: Dict[float, Dict[int, List[dict]]] = defaultdict(dict)
    for mpath in Path(sweep_root).glob("w*_s*/training_metrics.jsonl"):
        try:
            parent = mpath.parent.name  # 'w0.10_s2'
            weight = float(parent.split("_")[0][1:])
            seed = int(parent.split("_")[1][1:])
        except (ValueError, IndexError):
            continue
        records: List[dict] = []
        with mpath.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        out[weight][seed] = records
    return {w: dict(seeds) for w, seeds in out.items()}


def _summarize_one_run(records: List[dict]) -> dict:
    if not records:
        return {"n_iters": 0}
    rewards = np.array([float(r.get("episode_reward_mean", float("nan"))) for r in records])
    rewards = rewards[~np.isnan(rewards)]
    if rewards.size == 0:
        return {"n_iters": len(records), "no_rewards": True}

    first_pos_iter: Optional[int] = None
    for r in records:
        if float(r.get("episode_reward_mean", float("-inf"))) > 0:
            first_pos_iter = int(r.get("iteration", 0))
            break

    # Late-iteration stability: stddev over last 25% of training
    q = max(1, rewards.size // 4)
    late = rewards[-q:]
    late_std = float(np.std(late)) if late.size else float("nan")
    # Improvement trend over last 25%: slope sign
    if late.size >= 3:
        x = np.arange(late.size, dtype=float)
        slope = float(np.polyfit(x, late, 1)[0])
    else:
        slope = float("nan")

    return {
        "n_iters": len(records),
        "final_reward": float(rewards[-1]),
        "best_reward": float(rewards.max()),
        "worst_reward": float(rewards.min()),
        "first_positive_iter": first_pos_iter,
        "late_stddev": late_std,
        "late_slope": slope,
    }


def analyze(sweep_root: Path) -> str:
    runs = _load_metrics(Path(sweep_root))
    if not runs:
        return "## Training-curve diagnosis\n\n_(No training_metrics.jsonl files found.)_\n"

    lines: List[str] = []
    lines.append("## Training-curve diagnosis\n")
    lines.append(f"**Sweep:** `{sweep_root}`\n")

    # Per-(weight, seed) table
    lines.append("### Per-run convergence indicators\n")
    lines.append("| weight | seed | n_iters | final_reward | best_reward | first_positive | late_stddev | late_slope |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    per_weight_summaries: Dict[float, List[dict]] = defaultdict(list)
    for w in sorted(runs):
        for s in sorted(runs[w]):
            summary = _summarize_one_run(runs[w][s])
            per_weight_summaries[w].append(summary)
            if summary.get("n_iters", 0) == 0 or summary.get("no_rewards"):
                lines.append(f"| {w:.2f} | {s} | 0 | n/a | n/a | n/a | n/a | n/a |")
                continue
            first_pos = summary["first_positive_iter"]
            first_pos_str = str(first_pos) if first_pos is not None else "never"
            lines.append(
                f"| {w:.2f} | {s} | {summary['n_iters']} | "
                f"{summary['final_reward']:+.2f} | {summary['best_reward']:+.2f} | "
                f"{first_pos_str} | {summary['late_stddev']:.2f} | "
                f"{summary['late_slope']:+.3f} |"
            )
    lines.append("")

    # Per-weight aggregate
    lines.append("### Per-weight aggregate\n")
    lines.append("| weight | n_seeds | mean_final | stddev_final | mean_best | mean_late_slope |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    weight_overall: Dict[float, dict] = {}
    for w in sorted(per_weight_summaries):
        valid = [s for s in per_weight_summaries[w] if not s.get("no_rewards") and s.get("n_iters", 0) > 0]
        if not valid:
            lines.append(f"| {w:.2f} | 0 | n/a | n/a | n/a | n/a |")
            weight_overall[w] = {}
            continue
        finals = np.array([s["final_reward"] for s in valid])
        bests = np.array([s["best_reward"] for s in valid])
        slopes = np.array([s["late_slope"] for s in valid if not math.isnan(s["late_slope"])])
        mean_slope = float(slopes.mean()) if slopes.size else float("nan")
        weight_overall[w] = {
            "mean_final": float(finals.mean()),
            "stddev_final": float(finals.std(ddof=1)) if finals.size > 1 else float("nan"),
            "mean_best": float(bests.mean()),
            "mean_late_slope": mean_slope,
        }
        lines.append(
            f"| {w:.2f} | {len(valid)} | {weight_overall[w]['mean_final']:+.2f} | "
            f"{weight_overall[w]['stddev_final']:.2f} | "
            f"{weight_overall[w]['mean_best']:+.2f} | "
            f"{mean_slope:+.3f} |"
        )
    lines.append("")

    # Interpretation
    lines.append("### Interpretation\n")
    interp: List[str] = []
    # Cross-weight: did anything converge?
    overall_slopes = [w["mean_late_slope"] for w in weight_overall.values()
                      if w and not math.isnan(w.get("mean_late_slope", float("nan")))]
    if overall_slopes:
        median_slope = float(np.median(overall_slopes))
        if median_slope > 0.05:
            interp.append(
                f"- **Still improving at end of training**: median late-quartile slope = "
                f"{median_slope:+.3f}. Training was cut off before convergence; more steps "
                f"would likely change results."
            )
        elif median_slope < -0.05:
            interp.append(
                f"- **Decaying late in training**: median late-quartile slope = "
                f"{median_slope:+.3f}. Possible overfit, policy collapse, or unstable alpha schedule."
            )
        else:
            interp.append(
                f"- **Plateaued**: median late-quartile slope = {median_slope:+.3f} (near zero). "
                f"Training has converged; more steps unlikely to change much without other changes."
            )

    # Across-seed variance per weight
    stddevs = [w["stddev_final"] for w in weight_overall.values()
               if w and not math.isnan(w.get("stddev_final", float("nan")))]
    if stddevs:
        max_std = max(stddevs)
        if max_std > 5.0:
            interp.append(
                f"- **High across-seed variance**: max stddev_final = {max_std:.2f}. "
                f"3 seeds is too few; results are dominated by stochastic SAC noise. "
                f"Need 5-10 seeds per weight to detect real differences."
            )
        else:
            interp.append(
                f"- **Low across-seed variance**: max stddev_final = {max_std:.2f}. "
                f"Training is reproducible; differences between weights are real signal."
            )
    lines.extend(interp)
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("sweep_root", type=Path)
    args = p.parse_args()
    print(analyze(args.sweep_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
