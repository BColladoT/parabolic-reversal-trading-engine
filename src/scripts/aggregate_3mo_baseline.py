"""Aggregate 3 Discrete PPO seeds on the new 3-month OOS window.

Builds a sweep_summary.json-shaped dict from individual quick_test_results.json
files so the Phase 6 analyze_final.py path can consume it.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np


def _read_seed(d: Path) -> dict | None:
    p = d / "quick_test_results.json"
    if not p.exists():
        print(f"  WARNING: missing {p}", file=sys.stderr)
        return None
    r = json.loads(p.read_text())
    folds = r.get("folds") or []
    fm = folds[0] if folds else r.get("fold_metrics", r)
    test_pnl = fm.get("test_pnl_total", fm.get("total_test_pnl"))
    if test_pnl is None:
        print(f"  WARNING: no test_pnl in {p}", file=sys.stderr)
        return None
    return {
        "seed": int(d.name.rsplit("s", 1)[-1]),
        "test_pnl": float(test_pnl),
        "win_rate": fm.get("win_rate"),
        "mean_episode_pnl": fm.get("mean_episode_pnl"),
        "n_episodes": fm.get("test_episodes_evaluated"),
        "test_start": fm.get("test_start"),
        "test_end": fm.get("test_end"),
        "per_episode_results": fm.get("per_episode_results", []),
        "action_distribution": r.get("action_distribution"),
    }


def main():
    base = Path("models")
    seeds_dirs = sorted(base.glob("ppo_discrete_3mo_s*"))
    per_seed = [r for r in (_read_seed(d) for d in seeds_dirs) if r is not None]

    if not per_seed:
        print("ERROR: no seed results found", file=sys.stderr)
        sys.exit(1)

    pnls = [s["test_pnl"] for s in per_seed]
    mean_p = float(np.mean(pnls))
    median_p = float(np.median(pnls))
    std_p = float(np.std(pnls, ddof=1)) if len(pnls) >= 2 else None
    n_eps = [s["n_episodes"] for s in per_seed if s["n_episodes"] is not None]

    print("=" * 60)
    print("Discrete PPO 3-seed baseline on 2024-10-01 → 2024-12-30")
    print("=" * 60)
    for s in sorted(per_seed, key=lambda x: x["seed"]):
        print(f"  seed {s['seed']}: test_pnl=${s['test_pnl']:>8.0f}  win_rate={s['win_rate']:.3f}  n_episodes={s['n_episodes']}")
    print(f"\n  Mean:   ${mean_p:.0f}")
    print(f"  Median: ${median_p:.0f}")
    if std_p is not None:
        print(f"  Std:    ${std_p:.0f}")
    print(f"  Episodes evaluated per seed: {n_eps}")

    # Aggregate action distribution across seeds
    bin_keys = set()
    for s in per_seed:
        if s["action_distribution"]:
            bin_keys.update(int(k) for k in s["action_distribution"].keys())
    if bin_keys:
        avg_dist = {}
        for b in sorted(bin_keys):
            vals = [s["action_distribution"].get(str(b), s["action_distribution"].get(b, 0.0))
                    for s in per_seed if s["action_distribution"]]
            avg_dist[b] = float(np.mean(vals))
        print(f"\n  Avg action distribution across seeds:")
        for b in sorted(avg_dist):
            print(f"    bin {b}: {avg_dist[b]:.4f}")
        max_b, max_p = max(avg_dist.items(), key=lambda x: x[1])
        if max_p > 0.80:
            print(f"  WARNING: bin {max_b} dominates ({max_p:.1%}) → likely policy collapse")

    # Build sweep_summary.json shape for analyze_final.py compatibility
    summary = {
        "configs": [{
            "param": "discrete_action_bins",
            "value": "7",
            "per_seed_results": per_seed,
            "mean_test_pnl": mean_p,
            "median_test_pnl": median_p,
            "std_test_pnl": std_p,
            "n_failed": 0,
        }],
        "param": "discrete_action_bins",
        "values": ["7"],
        "seeds": [s["seed"] for s in per_seed],
        "methodology": {
            "test_start": "2024-10-01",
            "test_end": "2024-12-30",
            "n_setups_per_seed": n_eps,
        }
    }
    out = Path("reports/ppo_discrete_3mo_3seed_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n  Wrote summary: {out}")


if __name__ == "__main__":
    main()
