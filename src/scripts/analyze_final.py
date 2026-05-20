"""Phase 6 final analysis: 10-seed RL config vs verified rule baseline.

Reads a sweep_summary.json from ``run_sweep.py`` (typically a single-config,
N-seed sweep) and the Phase 0 rule baseline JSON, then computes the decision-
gate statistics for the final confirmation:

  * RL per-seed test_pnls (sorted), mean, median, std (ddof=1)
  * Bootstrap 95% CI on RL mean (delegates to
    ``src.utils.statistical_tests.bootstrap_confidence_interval``)
  * Rule baseline aggregate test_pnl, win_rate, mean_winner, mean_loser
  * Delta: RL_mean - rule_total
  * Action distribution + Shannon entropy (when present)
  * Verdict: SHIP_RL | PIVOT_TO_RULE | CONTINUE_RESEARCH

Verdict rules (higher PnL = better):
  * If RL 95% CI lower bound > 0 → SHIP_RL (RL is profitable with significance)
  * Else if rule_total > RL 95% CI upper → PIVOT_TO_RULE (rule statistically
    beats RL on aggregate test_pnl, regardless of profitability sign)
  * Else → CONTINUE_RESEARCH (the realistic "both negative, RL loses less,
    but not by a confidently large margin" case)

The decision NEVER says PIVOT just because rule is profitable — it must also
beat RL with statistical significance. This handles the "RL beat the rule by
$551 but both lost" scenario from the verified Phase 0 baseline.

Usage:
    python src/scripts/analyze_final.py \\
        --rl-sweep models/final_10seed_2026-05-20/sweep_summary.json \\
        --rule-baseline reports/rl_vs_rule_baseline_2026-05-20.json \\
        --output docs/discrete_ppo_final_stats.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Allow ``python src/scripts/analyze_final.py`` to import ``src.*`` without
# the caller having to set PYTHONPATH. Matches the pattern used by sibling
# scripts (train_wfo_quick_test.py, compare_rl_vs_rule.py, etc.).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the existing helper — DO NOT recreate.
from src.utils.statistical_tests import bootstrap_confidence_interval  # noqa: E402


def _extract_rule_baseline(rule_json: dict) -> dict:
    """Find the rule_baseline block, tolerating both shapes.

    The Phase 0 ``rl_vs_rule_baseline_*.json`` nests the block under
    ``summary_fixed_shares.rule_baseline``. A simpler hand-rolled JSON
    might put it at the top level. Accept either.
    """
    if "rule_baseline" in rule_json:
        return rule_json["rule_baseline"]
    if "summary_fixed_shares" in rule_json and "rule_baseline" in rule_json["summary_fixed_shares"]:
        return rule_json["summary_fixed_shares"]["rule_baseline"]
    raise KeyError(
        "Could not find rule_baseline block in rule JSON. Expected either "
        "top-level 'rule_baseline' or 'summary_fixed_shares.rule_baseline'."
    )


def _rule_total_pnl(rule_block: dict) -> float:
    """Extract aggregate test PnL. Tolerate 'total_test_pnl' or 'total_pnl'."""
    for k in ("total_test_pnl", "total_pnl"):
        if k in rule_block:
            return float(rule_block[k])
    raise KeyError(
        f"rule_baseline block missing total_test_pnl. Keys: {list(rule_block)}"
    )


def _action_entropy(action_dist: Optional[dict]) -> Optional[float]:
    """Shannon entropy of a discrete action distribution (nats).

    Returns None for empty / missing distributions. Zero-probability bins
    are skipped (0 * log 0 = 0). Negative probabilities raise.
    """
    if not action_dist:
        return None
    probs = []
    for v in action_dist.values():
        if v is None:
            continue
        v = float(v)
        if v < 0:
            raise ValueError(f"Negative probability in action_distribution: {v}")
        if v > 0:
            probs.append(v)
    if not probs:
        return None
    # Don't auto-normalize — caller is expected to pass probabilities. But
    # if the distribution sums to something other than ~1, fall back to
    # normalizing rather than emitting a misleading entropy value.
    total = sum(probs)
    if abs(total - 1.0) > 1e-3:
        probs = [p / total for p in probs]
    return float(-sum(p * np.log(p) for p in probs))


def _verdict(
    rl_ci_lo: float,
    rl_ci_hi: float,
    rl_mean: float,
    rule_total: float,
) -> dict:
    """Decision-gate logic. Returns {'verdict': str, 'reason': str}.

    Higher PnL = better. Rule beats RL when ``rule_total > rl_ci_hi``.
    """
    rule_significantly_beats_rl = rule_total > rl_ci_hi

    if rl_ci_lo > 0:
        return {
            "verdict": "SHIP_RL",
            "reason": (
                f"RL 95% CI lower bound (${rl_ci_lo:.0f}) > 0 - "
                f"RL is profitable with statistical significance."
            ),
        }
    if rule_significantly_beats_rl:
        if rule_total > 0:
            return {
                "verdict": "PIVOT_TO_RULE",
                "reason": (
                    f"Rule baseline (${rule_total:.0f}) is profitable AND "
                    f"statistically beats RL CI upper (${rl_ci_hi:.0f})."
                ),
            }
        return {
            "verdict": "PIVOT_TO_RULE",
            "reason": (
                f"Rule baseline (${rule_total:.0f}) statistically beats RL "
                f"CI upper (${rl_ci_hi:.0f}) - both lose, but rule loses "
                f"less with significance."
            ),
        }
    # CONTINUE: RL is not confidently profitable (rl_ci_lo <= 0) AND rule
    # does not statistically beat RL. Three sub-cases for clarity:
    rl_beats_rule_significantly = rule_total < rl_ci_lo
    if rl_beats_rule_significantly:
        # RL is below zero but its CI sits strictly above rule_total →
        # RL is BETTER than the rule with significance, but still loses.
        reason = (
            f"RL beats rule with significance "
            f"(rule=${rule_total:.0f} < RL CI lower=${rl_ci_lo:.0f}), but "
            f"RL itself is not yet profitable (CI lower <= 0). Keep tuning."
        )
        # Note: this is the most likely Phase 6 outcome given Phase 0 data
        # (rule -$2160, RL ~-$1610). It does NOT pivot to rule because rule
        # loses more; it does NOT ship RL because RL still loses money.
    elif rule_total < rl_mean:
        reason = (
            f"Both RL (mean=${rl_mean:.0f}, "
            f"CI=[${rl_ci_lo:.0f}, ${rl_ci_hi:.0f}]) and rule "
            f"(${rule_total:.0f}) lose. RL loses less on the point estimate, "
            f"but rule_total falls inside RL CI - the gap is not "
            f"statistically significant."
        )
    else:
        reason = (
            f"RL (mean=${rl_mean:.0f}, "
            f"CI=[${rl_ci_lo:.0f}, ${rl_ci_hi:.0f}]) does not clearly beat "
            f"rule (${rule_total:.0f}); RL CI lower is non-positive and "
            f"rule sits within or above RL CI."
        )
    return {"verdict": "CONTINUE_RESEARCH", "reason": reason}


def analyze(
    rl_summary: dict,
    rule_json: dict,
    *,
    config_label: str = "UNSPECIFIED",
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    bootstrap_seed: int = 42,
    config_index: int = 0,
) -> dict:
    """Compute the full stats dict.

    Splits cleanly from CLI so tests can call it without subprocess overhead.

    Args:
        rl_summary: parsed sweep_summary.json
        rule_json: parsed rule baseline JSON (Phase 0 shape)
        config_label: human-readable label for the headline
        n_bootstrap: bootstrap resample count
        confidence: e.g. 0.95
        bootstrap_seed: reproducibility seed
        config_index: which entry of rl_summary['configs'] to read (default 0;
            Phase 6 sweeps usually have a single config)
    """
    configs = rl_summary.get("configs", [])
    if not configs:
        raise ValueError("rl_summary has no 'configs' entries")
    if config_index >= len(configs):
        raise IndexError(
            f"config_index={config_index} out of range (n_configs={len(configs)})"
        )
    cfg = configs[config_index]

    # Filter to successful seeds: test_pnl present and non-None.
    per_seed = cfg.get("per_seed_results", [])
    rl_pnls = [
        float(r["test_pnl"])
        for r in per_seed
        if r.get("test_pnl") is not None
    ]
    n_seeds = len(rl_pnls)
    n_failed = sum(
        1 for r in per_seed if "error" in r or r.get("test_pnl") is None
    )

    if n_seeds < 3:
        raise ValueError(
            f"Only {n_seeds} successful seeds (n_failed={n_failed}). "
            f"Need at least 3 for bootstrap CI."
        )

    rl_mean = float(np.mean(rl_pnls))
    rl_median = float(np.median(rl_pnls))
    # ddof=1 = sample stddev (matches run_sweep.py convention).
    rl_std = float(np.std(rl_pnls, ddof=1))

    boot = bootstrap_confidence_interval(
        rl_pnls,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=bootstrap_seed,
    )
    rl_ci_lo = boot["ci_lower"]
    rl_ci_hi = boot["ci_upper"]

    rule_block = _extract_rule_baseline(rule_json)
    rule_total = _rule_total_pnl(rule_block)
    rule_win_rate = float(rule_block.get("win_rate", float("nan")))
    rule_mean_winner = float(rule_block.get("mean_winner", float("nan")))
    rule_mean_loser = float(rule_block.get("mean_loser", float("nan")))

    rule_significantly_beats_rl = rule_total > rl_ci_hi

    # Action distribution: pull from the FIRST seed that has it (Discrete
    # PPO writes action_distribution to its quick_test_results.json; the
    # sweep runner forwards it into per_seed_results[i]['action_distribution']).
    action_dist = None
    for r in per_seed:
        if r.get("action_distribution"):
            action_dist = r["action_distribution"]
            break

    action_entropy = _action_entropy(action_dist)
    n_bins = len(action_dist) if action_dist else 0
    action_entropy_max = float(np.log(n_bins)) if n_bins > 1 else None

    verdict = _verdict(rl_ci_lo, rl_ci_hi, rl_mean, rule_total)

    return {
        "config_label": config_label,
        "rl_sweep_param": cfg.get("param"),
        "rl_sweep_value": cfg.get("value"),
        "rl": {
            "n_seeds": n_seeds,
            "n_failed": n_failed,
            "per_seed_pnls_sorted": sorted(rl_pnls),
            "mean": rl_mean,
            "median": rl_median,
            "std": rl_std,
            "ci_95_lower": rl_ci_lo,
            "ci_95_upper": rl_ci_hi,
            "bootstrap_std_error": boot.get("std_error"),
            "action_distribution": action_dist,
            "action_entropy_nats": action_entropy,
            "action_entropy_max_nats": action_entropy_max,
        },
        "rule_baseline": {
            "total_test_pnl": rule_total,
            "win_rate": rule_win_rate,
            "mean_winner": rule_mean_winner,
            "mean_loser": rule_mean_loser,
        },
        "comparison": {
            "rl_mean_minus_rule_total": rl_mean - rule_total,
            "rule_significantly_beats_rl": rule_significantly_beats_rl,
        },
        **verdict,
    }


def format_report(stats: dict) -> str:
    """Render the stats dict as a console report. Pure function for testing."""
    rl = stats["rl"]
    rule = stats["rule_baseline"]
    cmp = stats["comparison"]
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append(f"Phase 6 Final Analysis: {stats['config_label']}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"  RL (Discrete PPO, {rl['n_seeds']} seeds, "
                 f"{rl['n_failed']} failed):")
    pnls_repr = [f"${p:.0f}" for p in rl["per_seed_pnls_sorted"]]
    lines.append(f"    Per-seed test_pnl (sorted): {pnls_repr}")
    lines.append(f"    Mean:    ${rl['mean']:.0f}")
    lines.append(f"    Median:  ${rl['median']:.0f}")
    lines.append(f"    Std:     ${rl['std']:.0f}")
    lines.append(
        f"    95% CI:  (${rl['ci_95_lower']:.0f}, ${rl['ci_95_upper']:.0f})"
    )
    if rl["action_entropy_nats"] is not None:
        max_h = rl["action_entropy_max_nats"]
        max_repr = f"{max_h:.3f}" if max_h is not None else "n/a"
        n_bins = len(rl["action_distribution"] or {})
        lines.append(
            f"    Action entropy: {rl['action_entropy_nats']:.3f} nats "
            f"(max for N={n_bins}: {max_repr})"
        )
    lines.append("")
    lines.append("  Rule baseline (deterministic):")
    lines.append(f"    Total test_pnl: ${rule['total_test_pnl']:.0f}")
    lines.append(f"    Win rate:       {rule['win_rate']:.1%}")
    lines.append(f"    Mean winner:    ${rule['mean_winner']:.0f}")
    lines.append(f"    Mean loser:     ${rule['mean_loser']:.0f}")
    lines.append("")
    lines.append("  Comparison:")
    lines.append(
        f"    RL mean - rule total:        "
        f"${cmp['rl_mean_minus_rule_total']:.0f}"
    )
    lines.append(
        f"    Rule significantly beats RL? "
        f"{cmp['rule_significantly_beats_rl']}"
    )
    lines.append("")
    lines.append(f"  VERDICT: {stats['verdict']}")
    lines.append(f"  Reason:  {stats['reason']}")
    lines.append("=" * 60)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 6 final analysis: 10-seed RL vs rule baseline."
    )
    ap.add_argument(
        "--rl-sweep", required=True,
        help="Path to sweep_summary.json from run_sweep.py.",
    )
    ap.add_argument(
        "--rule-baseline", required=True,
        help="Path to Phase 0 rule baseline JSON "
             "(e.g. reports/rl_vs_rule_baseline_2026-05-20.json).",
    )
    ap.add_argument(
        "--output", required=True,
        help="Output JSON path for the computed stats.",
    )
    ap.add_argument(
        "--config-label", default="UNSPECIFIED",
        help="Human-readable config label, e.g., "
             "'discrete_action_bins=7,total_steps=75000,entropy_coeff=0.01'.",
    )
    ap.add_argument(
        "--n-bootstrap", type=int, default=5000,
        help="Number of bootstrap resamples (default: 5000).",
    )
    ap.add_argument(
        "--config-index", type=int, default=0,
        help="Which sweep config entry to analyze (default: 0).",
    )
    args = ap.parse_args()

    rl_summary = json.loads(Path(args.rl_sweep).read_text())
    rule_json = json.loads(Path(args.rule_baseline).read_text())

    try:
        stats = analyze(
            rl_summary,
            rule_json,
            config_label=args.config_label,
            n_bootstrap=args.n_bootstrap,
            config_index=args.config_index,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2))

    print(format_report(stats))
    print(f"\nWrote stats to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
