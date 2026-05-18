"""Mask-violation diagnostic on a completed reward-weight sweep.

Parses each run.log for 'MASK VIOLATION' lines. Computes total violations,
violations per 1000 training steps (rough rate), and whether violation rate
decreased over training (split first vs second half of log).

Usage:
    python -m src.scripts.diagnose_masks models/sweep_2026-05-18/
"""
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

# Sample line we want to match:
# WARNING:src.rl.env:MASK VIOLATION #1: action_type=1 overridden to HOLD. mask=[0 0 1], VWAP dev=12.03, position=0.0
_VIOLATION_RE = re.compile(r"MASK VIOLATION #\d+: action_type=(\d)")
# Episode marker (used as proxy for training progress):
_EPISODE_RE = re.compile(r"\[TRAIN\] Episode reset")
# Target timesteps (from one of the script's INFO lines, e.g. "Timesteps: 25000"):
_TARGET_STEPS_RE = re.compile(r"Timesteps:\s*(\d+)")


def _parse_one_log(log_path: Path, target_steps_default: int = 25000) -> dict:
    """Parse a single run.log. Returns dict with counts + early/late split."""
    violations: List[int] = []  # action_type per violation, in order
    episode_marks: List[int] = []  # line number per episode reset
    target_steps = target_steps_default
    line_num = 0
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_num += 1
            m = _VIOLATION_RE.search(line)
            if m:
                violations.append(int(m.group(1)))
                continue
            if _EPISODE_RE.search(line):
                episode_marks.append(line_num)
                continue
            m2 = _TARGET_STEPS_RE.search(line)
            if m2:
                target_steps = int(m2.group(1))

    total_episodes = len(episode_marks)
    total = len(violations)

    # Split episodes into halves to compare early vs late violation rates.
    if total_episodes >= 4 and episode_marks:
        mid_line = episode_marks[total_episodes // 2]
        # Re-scan? No — we kept order. Find how many violations occurred before mid_line.
        # We need to know each violation's line number. Re-read for that.
        early_v = 0
        late_v = 0
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, 1):
                if _VIOLATION_RE.search(line):
                    if i < mid_line:
                        early_v += 1
                    else:
                        late_v += 1
        early_episodes = total_episodes // 2
        late_episodes = total_episodes - early_episodes
        early_rate = early_v / max(early_episodes, 1)
        late_rate = late_v / max(late_episodes, 1)
    else:
        early_v = late_v = 0
        early_rate = late_rate = float("nan")

    counts_per_action = Counter(violations)
    per_1000_steps = (total / max(target_steps, 1)) * 1000.0

    return {
        "total_violations": total,
        "total_episodes": total_episodes,
        "target_steps": target_steps,
        "per_1000_steps": per_1000_steps,
        "early_violations": early_v,
        "late_violations": late_v,
        "early_rate_per_episode": early_rate,
        "late_rate_per_episode": late_rate,
        "by_action_type": dict(counts_per_action),
    }


def _group_by_weight(sweep_root: Path) -> Dict[float, List[Path]]:
    out: Dict[float, List[Path]] = defaultdict(list)
    for log_path in Path(sweep_root).glob("w*_s*/run.log"):
        try:
            weight = float(log_path.parent.name.split("_")[0][1:])
        except (ValueError, IndexError):
            continue
        out[weight].append(log_path)
    return dict(out)


def analyze(sweep_root: Path) -> str:
    grouped = _group_by_weight(Path(sweep_root))
    if not grouped:
        return "## Mask-violation diagnosis\n\n_(No run.log files found.)_\n"

    lines: List[str] = []
    lines.append("## Mask-violation diagnosis\n")
    lines.append(f"**Sweep:** `{sweep_root}`\n")

    lines.append("### Per-run violation rates\n")
    lines.append("| weight | seed | total_viol | per_1000_steps | early_rate | late_rate | trend |")
    lines.append("|---|---:|---:|---:|---:|---:|:---:|")
    per_weight: Dict[float, List[dict]] = defaultdict(list)
    for w in sorted(grouped):
        for log_path in sorted(grouped[w]):
            seed = log_path.parent.name.split("_")[1][1:]
            s = _parse_one_log(log_path)
            per_weight[w].append(s)
            if s["early_rate_per_episode"] != s["early_rate_per_episode"]:  # NaN
                early_str = "n/a"
                late_str = "n/a"
                trend = "n/a"
            else:
                early_str = f"{s['early_rate_per_episode']:.2f}"
                late_str = f"{s['late_rate_per_episode']:.2f}"
                if s["late_rate_per_episode"] < s["early_rate_per_episode"] * 0.5:
                    trend = "drop"
                elif s["late_rate_per_episode"] > s["early_rate_per_episode"] * 1.5:
                    trend = "rise"
                else:
                    trend = "flat"
            lines.append(
                f"| {w:.2f} | {seed} | {s['total_violations']:,} | "
                f"{s['per_1000_steps']:.2f} | {early_str} | {late_str} | {trend} |"
            )
    lines.append("")

    # Aggregate
    lines.append("### Per-weight aggregate\n")
    lines.append("| weight | n_seeds | mean_per_1000_steps | mean_early | mean_late | learning? |")
    lines.append("|---|---:|---:|---:|---:|:---:|")
    weight_summaries = {}
    for w in sorted(per_weight):
        rs = per_weight[w]
        if not rs:
            continue
        per_1k = sum(s["per_1000_steps"] for s in rs) / len(rs)
        early_rates = [s["early_rate_per_episode"] for s in rs
                       if s["early_rate_per_episode"] == s["early_rate_per_episode"]]  # not NaN
        late_rates = [s["late_rate_per_episode"] for s in rs
                      if s["late_rate_per_episode"] == s["late_rate_per_episode"]]
        mean_early = sum(early_rates) / len(early_rates) if early_rates else float("nan")
        mean_late = sum(late_rates) / len(late_rates) if late_rates else float("nan")
        learning = "yes" if (mean_late == mean_late and mean_early == mean_early
                             and mean_late < mean_early * 0.5) else "no"
        weight_summaries[w] = {
            "per_1k": per_1k, "mean_early": mean_early, "mean_late": mean_late, "learning": learning,
        }
        early_disp = f"{mean_early:.2f}" if mean_early == mean_early else "n/a"
        late_disp = f"{mean_late:.2f}" if mean_late == mean_late else "n/a"
        lines.append(
            f"| {w:.2f} | {len(rs)} | {per_1k:.2f} | {early_disp} | {late_disp} | **{learning}** |"
        )
    lines.append("")

    # Interpretation
    lines.append("### Interpretation\n")
    interp: List[str] = []
    any_learning = any(s["learning"] == "yes" for s in weight_summaries.values())
    mean_rates = [s["per_1k"] for s in weight_summaries.values()]
    overall_rate = sum(mean_rates) / len(mean_rates) if mean_rates else 0.0
    if not any_learning:
        interp.append(
            f"- **Mask not being learned**: no weight shows a clear drop in violation rate "
            f"from early to late training. Mean rate is ~{overall_rate:.1f} per 1000 steps. "
            f"The policy keeps trying invalid actions throughout training, which means the "
            f"reward signal isn't successfully shaping the gradient toward valid regions. "
            f"This is structural — more training won't fix it."
        )
    else:
        learners = [f"{w:.2f}" for w, s in weight_summaries.items() if s["learning"] == "yes"]
        interp.append(
            f"- **Mask partially learned**: weights {{{', '.join(learners)}}} show violation "
            f"rate dropping by 50%+ from early to late training. The policy adapts."
        )

    if overall_rate > 100:
        interp.append(
            f"- **Very high violation rate** ({overall_rate:.0f} per 1000 steps) suggests "
            f"the agent is hitting the mask on most steps. Either the policy parameterization "
            f"is incompatible with the mask shape, or the masking penalty isn't strong enough "
            f"to deflect exploration away from invalid regions."
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
