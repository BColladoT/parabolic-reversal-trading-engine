"""Unit tests for ``src/scripts/analyze_final.py``.

Imports the module's pure functions directly (no subprocess) — faster and
deterministic. The CLI is exercised once via subprocess to confirm wiring.

Cases covered:
  1. RL clearly wins (positive CI lower) → SHIP_RL
  2. RL clearly loses to rule (rule_total > RL CI upper) → PIVOT_TO_RULE
  3. Both negative, RL beats rule but not significantly → CONTINUE_RESEARCH
     (the realistic scenario per Phase 0 verified data)
  4. Empty / failed seeds → graceful error
  5. Bootstrap CI lower < mean < upper, both floats
  6. Action-entropy edge cases (None, all-zero, valid)
  7. End-to-end CLI smoke
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# conftest.py inserts repo root on sys.path before any 'src.*' import.
from src.scripts.analyze_final import (
    _action_entropy,
    _extract_rule_baseline,
    _rule_total_pnl,
    _verdict,
    analyze,
    format_report,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _make_sweep(pnls, *, param="discrete_action_bins", value="7",
                action_dist=None):
    """Synthesize a sweep_summary.json payload from per-seed PnLs."""
    per_seed = []
    for i, pnl in enumerate(pnls):
        entry = {
            "seed": 42 + i,
            "test_pnl": pnl,
            "win_rate": 0.3,
            "wall_time_s": 1.0,
        }
        if pnl is None:
            entry["error"] = "fake failure"
        if action_dist is not None and i == 0:
            entry["action_distribution"] = action_dist
        per_seed.append(entry)
    successful = [p for p in pnls if p is not None]
    return {
        "configs": [
            {
                "param": param,
                "value": value,
                "per_seed_results": per_seed,
                "mean_test_pnl": (
                    float(np.mean(successful)) if successful else None
                ),
                "median_test_pnl": (
                    float(np.median(successful)) if successful else None
                ),
                "std_test_pnl": (
                    float(np.std(successful, ddof=1))
                    if len(successful) >= 2 else None
                ),
                "n_failed": sum(1 for p in pnls if p is None),
            },
        ],
        "param": param,
        "values": [value],
        "seeds": list(range(42, 42 + len(pnls))),
        "algo": "ppo",
        "action_space": "discrete",
        "total_steps": "75000",
        "dry_run": False,
    }


def _make_rule_json(total, *, win_rate=0.357, mean_winner=51.0,
                    mean_loser=-302.0, nested=True):
    """Synthesize a rule baseline JSON. Default nests under
    ``summary_fixed_shares`` to match Phase 0 shape."""
    block = {
        "total_test_pnl": total,
        "win_rate": win_rate,
        "mean_winner": mean_winner,
        "mean_loser": mean_loser,
        "n_winners": 5,
        "n_losers": 8,
        "n_zero_pnl": 1,
    }
    if nested:
        return {"summary_fixed_shares": {"rule_baseline": block}}
    return {"rule_baseline": block}


# ---------------------------------------------------------------------------
# Verdict logic — three branches
# ---------------------------------------------------------------------------

def test_verdict_ship_rl_when_ci_lower_positive():
    out = _verdict(rl_ci_lo=100.0, rl_ci_hi=500.0, rl_mean=300.0,
                   rule_total=-2160.0)
    assert out["verdict"] == "SHIP_RL"
    assert "CI lower" in out["reason"]


def test_verdict_pivot_when_rule_profitable_and_beats_rl():
    # Rule is profitable AND above RL CI upper → unambiguous pivot.
    out = _verdict(rl_ci_lo=-500.0, rl_ci_hi=100.0, rl_mean=-200.0,
                   rule_total=500.0)
    assert out["verdict"] == "PIVOT_TO_RULE"
    assert "profitable" in out["reason"]


def test_verdict_pivot_when_both_negative_but_rule_beats_rl_significantly():
    # Rule loses less AND its total is above RL CI upper.
    out = _verdict(rl_ci_lo=-3000.0, rl_ci_hi=-1500.0, rl_mean=-2200.0,
                   rule_total=-1000.0)
    assert out["verdict"] == "PIVOT_TO_RULE"
    assert "both lose" in out["reason"]


def test_verdict_continue_when_rl_beats_rule_significantly_but_unprofitable():
    """Phase 0 realistic scenario: rule = -$2160, RL CI = [-2000, -1200].

    Rule is below RL CI lower → RL beats rule with significance, BUT RL
    itself is still losing money (CI lower <= 0). Verdict: continue tuning.
    """
    out = _verdict(rl_ci_lo=-2000.0, rl_ci_hi=-1200.0, rl_mean=-1610.0,
                   rule_total=-2160.0)
    assert out["verdict"] == "CONTINUE_RESEARCH"
    assert "RL beats rule" in out["reason"]
    assert "not yet profitable" in out["reason"]


def test_verdict_continue_when_both_lose_rule_inside_ci_below_mean():
    """Both lose, rule inside RL's CI but below RL mean.

    RL is better on point estimate but rule_total is not below RL CI lower,
    so the gap is not statistically significant.
    """
    # rl_ci_lo=-2500, rl_ci_hi=-1500, rl_mean=-2000; rule=-2200 sits inside CI
    # and is below rl_mean.
    out = _verdict(rl_ci_lo=-2500.0, rl_ci_hi=-1500.0, rl_mean=-2000.0,
                   rule_total=-2200.0)
    assert out["verdict"] == "CONTINUE_RESEARCH"
    assert "not statistically significant" in out["reason"]


def test_verdict_continue_when_rule_above_rl_mean_but_inside_ci():
    """Rule sits inside RL's CI but above RL mean - point estimate favors rule."""
    out = _verdict(rl_ci_lo=-2500.0, rl_ci_hi=-1500.0, rl_mean=-2000.0,
                   rule_total=-1800.0)
    assert out["verdict"] == "CONTINUE_RESEARCH"
    assert "does not clearly beat" in out["reason"]


def test_verdict_continue_when_cis_overlap():
    # Rule sits inside RL's CI - nothing is statistically significant.
    out = _verdict(rl_ci_lo=-2500.0, rl_ci_hi=-1500.0, rl_mean=-2000.0,
                   rule_total=-2000.0)
    assert out["verdict"] == "CONTINUE_RESEARCH"


# ---------------------------------------------------------------------------
# Rule-baseline extraction — accepts both shapes
# ---------------------------------------------------------------------------

def test_extract_rule_baseline_nested():
    rb = _extract_rule_baseline(_make_rule_json(-2160.0, nested=True))
    assert rb["total_test_pnl"] == -2160.0


def test_extract_rule_baseline_top_level():
    rb = _extract_rule_baseline(_make_rule_json(-2160.0, nested=False))
    assert rb["total_test_pnl"] == -2160.0


def test_extract_rule_baseline_missing_raises():
    with pytest.raises(KeyError, match="rule_baseline"):
        _extract_rule_baseline({"foo": "bar"})


def test_rule_total_pnl_missing_raises():
    with pytest.raises(KeyError, match="total_test_pnl"):
        _rule_total_pnl({"win_rate": 0.5})


# ---------------------------------------------------------------------------
# Action entropy edge cases
# ---------------------------------------------------------------------------

def test_action_entropy_none_returns_none():
    assert _action_entropy(None) is None


def test_action_entropy_empty_returns_none():
    assert _action_entropy({}) is None


def test_action_entropy_uniform_equals_log_n():
    # Uniform over 7 bins → H = log(7).
    dist = {str(i): 1.0 / 7 for i in range(7)}
    h = _action_entropy(dist)
    assert h == pytest.approx(np.log(7), rel=1e-6)


def test_action_entropy_collapsed_near_zero():
    # 99% in one bin → entropy small.
    dist = {"0": 0.99, "1": 0.0025, "2": 0.0025, "3": 0.0025, "4": 0.0025}
    h = _action_entropy(dist)
    assert 0.0 < h < 0.1


def test_action_entropy_handles_zero_probs():
    # Zero bins are skipped — should not produce nan/inf.
    dist = {"0": 0.5, "1": 0.5, "2": 0.0, "3": 0.0}
    h = _action_entropy(dist)
    assert h == pytest.approx(np.log(2), rel=1e-6)


def test_action_entropy_negative_prob_raises():
    with pytest.raises(ValueError, match="Negative probability"):
        _action_entropy({"0": -0.1, "1": 1.1})


def test_action_entropy_normalizes_unnormalized():
    # Counts instead of probs → should still produce log(2) for 50/50.
    dist = {"0": 50.0, "1": 50.0}
    h = _action_entropy(dist)
    assert h == pytest.approx(np.log(2), rel=1e-6)


# ---------------------------------------------------------------------------
# End-to-end: analyze() returns the expected shape & values
# ---------------------------------------------------------------------------

def test_analyze_continue_research_realistic_scenario():
    """Phase 0 realistic: 10 seeds around -$1600, rule at -$2160."""
    rng = np.random.default_rng(0)
    pnls = (-1600.0 + rng.normal(0, 200, size=10)).tolist()
    sweep = _make_sweep(
        pnls,
        action_dist={"0": 0.4, "1": 0.15, "2": 0.15, "3": 0.1, "4": 0.1,
                     "5": 0.05, "6": 0.05},
    )
    rule = _make_rule_json(-2160.19)
    stats = analyze(sweep, rule, config_label="phase6-test")

    assert stats["verdict"] == "CONTINUE_RESEARCH"
    assert stats["rl"]["n_seeds"] == 10
    assert stats["rl"]["n_failed"] == 0
    assert len(stats["rl"]["per_seed_pnls_sorted"]) == 10
    # Sorted ascending.
    sorted_pnls = stats["rl"]["per_seed_pnls_sorted"]
    assert sorted_pnls == sorted(sorted_pnls)
    # CI bracketing.
    assert stats["rl"]["ci_95_lower"] < stats["rl"]["mean"] < stats["rl"]["ci_95_upper"]
    assert isinstance(stats["rl"]["ci_95_lower"], float)
    assert isinstance(stats["rl"]["ci_95_upper"], float)
    # Action entropy populated.
    assert stats["rl"]["action_entropy_nats"] is not None
    assert stats["rl"]["action_entropy_max_nats"] == pytest.approx(np.log(7))
    # Rule fields propagated.
    assert stats["rule_baseline"]["total_test_pnl"] == pytest.approx(-2160.19)
    # Delta = RL_mean - rule_total > 0 (RL loses less).
    assert stats["comparison"]["rl_mean_minus_rule_total"] > 0


def test_analyze_ship_rl_when_rl_clearly_profitable():
    """All 10 seeds positive, tight cluster → CI lower > 0."""
    pnls = [500.0 + i * 10 for i in range(10)]
    sweep = _make_sweep(pnls)
    rule = _make_rule_json(-2160.0)
    stats = analyze(sweep, rule)
    assert stats["verdict"] == "SHIP_RL"
    assert stats["rl"]["ci_95_lower"] > 0


def test_analyze_pivot_when_rule_clearly_beats_rl():
    """RL clusters around -$3000, rule at -$500 → rule above RL CI upper."""
    rng = np.random.default_rng(1)
    pnls = (-3000.0 + rng.normal(0, 150, size=10)).tolist()
    sweep = _make_sweep(pnls)
    rule = _make_rule_json(-500.0)
    stats = analyze(sweep, rule)
    assert stats["verdict"] == "PIVOT_TO_RULE"
    assert stats["rule_baseline"]["total_test_pnl"] > stats["rl"]["ci_95_upper"]


# ---------------------------------------------------------------------------
# Graceful failure paths
# ---------------------------------------------------------------------------

def test_analyze_raises_when_too_few_seeds():
    sweep = _make_sweep([100.0, 200.0])  # Only 2 seeds.
    rule = _make_rule_json(-2160.0)
    with pytest.raises(ValueError, match="successful seeds"):
        analyze(sweep, rule)


def test_analyze_raises_when_all_seeds_failed():
    sweep = _make_sweep([None, None, None, None, None])
    rule = _make_rule_json(-2160.0)
    with pytest.raises(ValueError, match="successful seeds"):
        analyze(sweep, rule)


def test_analyze_handles_mixed_success_and_failure():
    pnls = [-1500.0, -1700.0, None, -1600.0, -1800.0, -1400.0]
    sweep = _make_sweep(pnls)
    rule = _make_rule_json(-2160.0)
    stats = analyze(sweep, rule)
    assert stats["rl"]["n_seeds"] == 5
    assert stats["rl"]["n_failed"] == 1


def test_analyze_raises_on_empty_configs():
    sweep = {"configs": [], "param": "x", "values": [], "seeds": []}
    rule = _make_rule_json(-2160.0)
    with pytest.raises(ValueError, match="no 'configs'"):
        analyze(sweep, rule)


def test_analyze_raises_on_bad_config_index():
    sweep = _make_sweep([100.0, 200.0, 300.0])
    rule = _make_rule_json(-2160.0)
    with pytest.raises(IndexError):
        analyze(sweep, rule, config_index=5)


# ---------------------------------------------------------------------------
# Bootstrap CI shape & properties
# ---------------------------------------------------------------------------

def test_bootstrap_ci_is_tuple_of_floats_with_lower_lt_upper():
    pnls = [-1600.0 + 100 * i for i in range(10)]
    sweep = _make_sweep(pnls)
    rule = _make_rule_json(-2160.0)
    stats = analyze(sweep, rule, n_bootstrap=2000)
    lo = stats["rl"]["ci_95_lower"]
    hi = stats["rl"]["ci_95_upper"]
    assert isinstance(lo, float)
    assert isinstance(hi, float)
    assert lo < hi


def test_bootstrap_ci_is_deterministic_across_calls():
    """Seed=42 in bootstrap_confidence_interval should give the same CI."""
    pnls = [-1500.0, -1700.0, -1600.0, -1800.0, -1400.0,
            -1550.0, -1650.0, -1750.0, -1450.0, -1500.0]
    sweep = _make_sweep(pnls)
    rule = _make_rule_json(-2160.0)
    s1 = analyze(sweep, rule, n_bootstrap=2000, bootstrap_seed=42)
    s2 = analyze(sweep, rule, n_bootstrap=2000, bootstrap_seed=42)
    assert s1["rl"]["ci_95_lower"] == s2["rl"]["ci_95_lower"]
    assert s1["rl"]["ci_95_upper"] == s2["rl"]["ci_95_upper"]


# ---------------------------------------------------------------------------
# format_report — sanity-check output rendering
# ---------------------------------------------------------------------------

def test_format_report_contains_key_lines():
    pnls = [-1600.0 + 100 * i for i in range(10)]
    sweep = _make_sweep(
        pnls,
        action_dist={"0": 0.4, "1": 0.1, "2": 0.1, "3": 0.1, "4": 0.1,
                     "5": 0.1, "6": 0.1},
    )
    rule = _make_rule_json(-2160.0)
    stats = analyze(sweep, rule, config_label="test-config")
    report = format_report(stats)
    assert "test-config" in report
    assert "VERDICT" in report
    assert "Per-seed test_pnl" in report
    assert "Mean:" in report
    assert "95% CI" in report
    assert "Action entropy" in report


# ---------------------------------------------------------------------------
# CLI smoke test — exercises full path, including arg parsing and file IO.
# ---------------------------------------------------------------------------

def test_cli_smoke(tmp_path):
    sweep_path = tmp_path / "sweep_summary.json"
    rule_path = tmp_path / "rule.json"
    out_path = tmp_path / "stats.json"

    pnls = [-1600.0 + 100 * i for i in range(10)]
    sweep_path.write_text(json.dumps(_make_sweep(pnls)))
    rule_path.write_text(json.dumps(_make_rule_json(-2160.0)))

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "src" / "scripts" / "analyze_final.py"),
            "--rl-sweep", str(sweep_path),
            "--rule-baseline", str(rule_path),
            "--output", str(out_path),
            "--config-label", "smoke-test",
            "--n-bootstrap", "1000",
        ],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        timeout=60,
    )
    assert result.returncode == 0, (
        f"CLI failed (rc={result.returncode}).\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert out_path.exists()
    stats = json.loads(out_path.read_text())
    assert stats["config_label"] == "smoke-test"
    assert stats["rl"]["n_seeds"] == 10
    assert stats["verdict"] in {"SHIP_RL", "PIVOT_TO_RULE", "CONTINUE_RESEARCH"}
    assert "VERDICT" in result.stdout
