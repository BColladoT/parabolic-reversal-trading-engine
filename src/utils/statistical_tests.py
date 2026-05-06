"""
Statistical tests for evaluating RL agent performance.

Provides:
1. Bootstrap confidence interval — error bars on PnL estimates
2. Monte Carlo permutation test — p-value for "RL beats baseline"

These are evaluation-time tools run AFTER training to validate
whether the agent has learned useful behavior or just got lucky.
"""

import numpy as np
from typing import List, Dict, Any


def bootstrap_confidence_interval(
    pnls: List[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Compute bootstrap confidence interval for mean PnL.

    Resamples the per-episode PnLs with replacement to estimate
    the sampling distribution of the mean. Returns the CI bounds
    and the standard error.

    Args:
        pnls: Per-episode PnL values
        n_bootstrap: Number of bootstrap resamples
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        seed: Random seed for reproducibility

    Returns:
        Dict with 'mean', 'ci_lower', 'ci_upper', 'std_error',
        'confidence_level', 'n_episodes', 'n_bootstrap'
    """
    rng = np.random.RandomState(seed)
    pnls_arr = np.array(pnls, dtype=np.float64)
    n = len(pnls_arr)

    if n < 2:
        return {
            'mean': float(pnls_arr.mean()) if n > 0 else 0.0,
            'ci_lower': None,
            'ci_upper': None,
            'std_error': None,
            'confidence_level': confidence,
            'n_episodes': n,
            'n_bootstrap': 0
        }

    # Vectorized bootstrap resampling
    indices = rng.randint(0, n, size=(n_bootstrap, n))
    boot_means = pnls_arr[indices].mean(axis=1)

    alpha = 1 - confidence
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return {
        'mean': float(pnls_arr.mean()),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'std_error': float(boot_means.std()),
        'confidence_level': confidence,
        'n_episodes': n,
        'n_bootstrap': n_bootstrap
    }


def permutation_test(
    rl_pnls: List[float],
    baseline_pnls: List[float],
    n_permutations: int = 10000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Monte Carlo permutation test: Is RL significantly better than baseline?

    For each episode, the RL and baseline produced different PnLs on the
    SAME setup. Under the null hypothesis (no difference), the assignment
    of "RL" vs "baseline" label to each pair of PnLs is arbitrary.

    We randomly swap RL/baseline labels for each episode, recompute the
    mean PnL difference, and build a null distribution. The p-value is
    the fraction of permutations where the shuffled difference exceeds
    the observed difference.

    Args:
        rl_pnls: Per-episode PnL from RL agent
        baseline_pnls: Per-episode PnL from baseline agent (same episodes)
        n_permutations: Number of random permutations
        seed: Random seed

    Returns:
        Dict with 'observed_diff', 'p_value', 'significant_at_05',
        'significant_at_01', 'null_mean', 'null_std'
    """
    rng = np.random.RandomState(seed)
    rl_arr = np.array(rl_pnls, dtype=np.float64)
    base_arr = np.array(baseline_pnls, dtype=np.float64)

    assert len(rl_arr) == len(base_arr), (
        f"Must have same number of episodes: RL={len(rl_arr)}, baseline={len(base_arr)}"
    )
    n = len(rl_arr)

    if n == 0:
        return {
            'observed_diff': 0.0,
            'p_value': 1.0,
            'significant_at_05': False,
            'significant_at_01': False,
            'null_mean': 0.0,
            'null_std': 0.0,
            'n_permutations': 0,
            'n_episodes': 0
        }

    # Observed test statistic: mean PnL difference
    diffs = rl_arr - base_arr
    observed_diff = float(diffs.mean())

    # Vectorized null distribution: randomly flip signs of differences
    # Equivalent to swapping RL/baseline labels per episode
    signs = rng.choice([-1, 1], size=(n_permutations, n))
    null_diffs = (signs * diffs).mean(axis=1)

    # One-sided p-value: P(null_diff >= observed_diff)
    p_value = float((null_diffs >= observed_diff).mean())

    return {
        'observed_diff': observed_diff,
        'p_value': p_value,
        'significant_at_05': p_value < 0.05,
        'significant_at_01': p_value < 0.01,
        'null_mean': float(null_diffs.mean()),
        'null_std': float(null_diffs.std()),
        'n_permutations': n_permutations,
        'n_episodes': n
    }


def format_benchmark_report(
    rl_pnls: List[float],
    baseline_results: Dict[str, List[float]],
    permutation_results: Dict[str, Dict[str, Any]] = None
) -> str:
    """
    Format a human-readable benchmark comparison report.

    Args:
        rl_pnls: RL agent per-episode PnLs
        baseline_results: {name: per_episode_pnls} for each baseline
        permutation_results: {name: permutation_test_result} (optional)

    Returns:
        Formatted string report
    """
    lines = []
    lines.append("=" * 70)
    lines.append("STATISTICAL BENCHMARKS")
    lines.append("=" * 70)

    # RL agent with bootstrap CI
    ci = bootstrap_confidence_interval(rl_pnls)
    rl_total = sum(rl_pnls)
    if ci['ci_lower'] is not None:
        lines.append(
            f"  RL Agent:       ${rl_total:>10,.2f} total | "
            f"mean ${ci['mean']:>8,.2f} "
            f"(95% CI: [${ci['ci_lower']:,.2f}, ${ci['ci_upper']:,.2f}])"
        )
    else:
        lines.append(f"  RL Agent:       ${rl_total:>10,.2f} total")

    # Each baseline
    for name, pnls in baseline_results.items():
        total = sum(pnls)
        mean = np.mean(pnls) if pnls else 0.0
        lines.append(f"  {name + ':':16s}${total:>10,.2f} total | mean ${mean:>8,.2f}")

    lines.append("")

    # Permutation tests
    if permutation_results:
        lines.append("  Significance tests (RL vs baseline):")
        for name, result in permutation_results.items():
            p = result['p_value']
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "n.s."
            lines.append(
                f"    vs {name + ':':14s}diff=${result['observed_diff']:>8,.2f}/ep, "
                f"p={p:.4f} {sig}"
            )
        lines.append("")
        lines.append("  Key: *** p<0.01, ** p<0.05, * p<0.10, n.s. = not significant")

    lines.append("=" * 70)
    return "\n".join(lines)
