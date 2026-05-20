"""
RL vs Rule Baseline Comparison Runner

This script runs the rule baseline on the same WFO folds as RL
and generates a direct comparison report with verdicts.

Usage:
    python -m src.scripts.compare_rl_vs_rule \
        --rl-results models/wfo/wfo_results.json \
        --output reports/rl_vs_rule_comparison.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.baselines.rule_baseline import (
    create_v5_relaxed_agent_fixed_shares,
    create_v5_relaxed_agent_fraction_of_equity
)
from src.baselines.random_agent import RandomAgent
from src.baselines.naive_short_agent import NaiveShortAgent
from src.baselines.evaluate_baseline import evaluate_baseline_on_fold
from src.utils.metrics import compute_fold_metrics
from src.utils.statistical_tests import (
    bootstrap_confidence_interval,
    permutation_test,
    format_benchmark_report
)


def load_rl_results(path: str) -> Dict[str, Any]:
    """Load RL results from train_wfo.py output."""
    with open(path, 'r') as f:
        return json.load(f)


def _extract_test_setups(fold_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract test setups from a fold result, supporting both JSON shapes.

    The quick-test runner (`train_wfo_quick_test.py`) writes
    `fold['per_episode_results']` directly. The production WFO runner
    (`train_wfo.py`) nests them under `fold['test_metrics']['per_episode_results']`.
    """
    per_episode = (
        fold_data.get('per_episode_results')
        or fold_data.get('test_metrics', {}).get('per_episode_results')
        or []
    )
    return [{'symbol': ep['symbol'], 'date': ep['date']} for ep in per_episode]


def _winner_loser_stats(pnls: List[float]) -> Dict[str, Any]:
    """
    Compute mean_winner, mean_loser, and zero-trade counts from a list of
    per-setup PnLs. Used to enrich the headline output beyond what
    ``compute_fold_metrics`` returns by default.

    Returns a dict with: mean_winner (or None if no winners), mean_loser
    (or None if no losers), n_winners, n_losers, n_zero.
    """
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]
    zeros = [p for p in pnls if p == 0]
    return {
        'mean_winner': float(np.mean(winners)) if winners else None,
        'mean_loser': float(np.mean(losers)) if losers else None,
        'n_winners': len(winners),
        'n_losers': len(losers),
        'n_zero_pnl': len(zeros),
    }


def compare_fold_metrics(
    rl_metrics: Dict[str, Any],
    rule_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare RL metrics to rule baseline metrics.
    
    Returns comparison with difference, ratio, and verdict.
    """
    if rl_metrics.get('episodes_evaluated', 0) == 0:
        return {'error': 'RL has no valid evaluations'}
    if rule_metrics.get('episodes_evaluated', 0) == 0:
        return {'error': 'Rule baseline has no valid evaluations'}
    
    # Extract key metrics
    rl_pnl = rl_metrics['total_test_pnl']
    rule_pnl = rule_metrics['total_test_pnl']
    
    # Calculate improvement percentage
    if rule_pnl != 0:
        pnl_improvement_pct = ((rl_pnl - rule_pnl) / abs(rule_pnl)) * 100
    else:
        pnl_improvement_pct = float('inf') if rl_pnl > 0 else float('-inf') if rl_pnl < 0 else 0
    
    # Determine verdict
    if pnl_improvement_pct >= 10:
        verdict = "PASS"
        verdict_reason = f"RL beats rule by {pnl_improvement_pct:.1f}% (>= 10% threshold)"
    elif pnl_improvement_pct >= 0:
        verdict = "MARGINAL"
        verdict_reason = f"RL beats rule by {pnl_improvement_pct:.1f}% (0-10% margin)"
    else:
        verdict = "FAIL"
        verdict_reason = f"RL underperforms rule by {abs(pnl_improvement_pct):.1f}%"
    
    return {
        'rl_total_pnl': rl_pnl,
        'rule_total_pnl': rule_pnl,
        'pnl_difference': rl_pnl - rule_pnl,
        'pnl_improvement_pct': pnl_improvement_pct,
        'pnl_ratio': rl_pnl / rule_pnl if rule_pnl != 0 else None,
        'rl_win_rate': rl_metrics['win_rate'],
        'rule_win_rate': rule_metrics['win_rate'],
        'rl_mean_pnl': rl_metrics['mean_episode_pnl'],
        'rule_mean_pnl': rule_metrics['mean_episode_pnl'],
        'rl_total_trades': rl_metrics['total_trades'],
        'rule_total_trades': rule_metrics['total_trades'],
        'rl_avg_trades_per_ep': rl_metrics['mean_trades_per_episode'],
        'rule_avg_trades_per_ep': rule_metrics['mean_trades_per_episode'],
        'verdict': verdict,
        'verdict_reason': verdict_reason
    }


def run_rule_baseline_on_fold(
    fold_data: Dict[str, Any],
    sizing_mode: str = "fixed_shares",
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run rule baseline on the same test setups as a fold of RL results.
    
    Args:
        fold_data: The per-fold result from RL (contains test dates)
        sizing_mode: "fixed_shares" or "fixed_fraction_of_equity"
        verbose: Print progress
        
    Returns:
        Metrics dict for the rule baseline
    """
    # Create agent based on sizing mode
    if sizing_mode == "fixed_shares":
        agent = create_v5_relaxed_agent_fixed_shares(shares=100)
    elif sizing_mode == "fixed_fraction_of_equity":
        agent = create_v5_relaxed_agent_fraction_of_equity(
            fraction=0.30,
            max_position=30000.0
        )
    else:
        raise ValueError(f"Unknown sizing mode: {sizing_mode}")
    
    # Build test setups from RL results.
    # Quick-test JSONs (quick_test_results.json) store per_episode_results at
    # the fold level. Full WFO JSONs (wfo_results.json) wrap it inside
    # fold['test_metrics']. Support both shapes so the harness works against
    # any train_wfo* output without rerunning training.
    test_setups = _extract_test_setups(fold_data)

    if len(test_setups) == 0:
        if verbose:
            print(f"  No test setups found for fold {fold_data.get('fold', '?')}")
        return compute_fold_metrics([], 0, 0)
    
    # Build env config for test window
    test_start = fold_data.get('test_start', '')
    test_end = fold_data.get('test_end', '')
    
    env_config = {
        "initial_capital": 100000.0,
        "date_range": (test_start[:10], test_end[:10]) if test_start and test_end else None,
        "mode": "eval"
    }
    
    if verbose:
        print(f"  Running rule baseline with {sizing_mode} sizing...")
        print(f"  Test setups: {len(test_setups)}")
    
    # Evaluate
    metrics = evaluate_baseline_on_fold(
        agent=agent,
        test_setups=test_setups,
        env_config=env_config,
        verbose=verbose
    )
    
    return metrics


def generate_comparison_report(
    rl_results: Dict[str, Any],
    rule_fixed_shares_results: List[Dict],
    rule_fraction_results: List[Dict]
) -> Dict[str, Any]:
    """
    Generate full comparison report.
    
    Returns report dict with per-fold and aggregate comparisons.
    """
    report = {
        'generated_at': datetime.now().isoformat(),
        'comparison_methodology': 'Same test setups, same execution/accounting',
        'sizing_modes_compared': ['fixed_shares', 'fixed_fraction_of_equity'],
        'per_fold': [],
        'aggregate': {}
    }
    
    rl_folds = rl_results.get('per_fold_results', rl_results.get('folds', []))
    
    for i, rl_fold in enumerate(rl_folds):
        fold_num = rl_fold.get('fold', i + 1)
        
        # Get rule results for this fold
        rule_fs = rule_fixed_shares_results[i] if i < len(rule_fixed_shares_results) else None
        rule_frac = rule_fraction_results[i] if i < len(rule_fraction_results) else None
        
        # RL metrics
        rl_metrics = rl_fold.get('test_metrics', rl_fold)
        
        fold_comparison = {
            'fold': fold_num,
            'test_window': {
                'start': rl_fold.get('test_start', ''),
                'end': rl_fold.get('test_end', '')
            },
            'rl_metrics': {
                'total_test_pnl': rl_metrics.get('total_test_pnl'),
                'mean_episode_pnl': rl_metrics.get('mean_episode_pnl'),
                'win_rate': rl_metrics.get('win_rate'),
                'total_trades': rl_metrics.get('total_trades'),
                'mean_trades_per_episode': rl_metrics.get('mean_trades_per_episode')
            }
        }
        
        # Compare vs fixed shares
        if rule_fs:
            fold_comparison['vs_fixed_shares'] = compare_fold_metrics(rl_metrics, rule_fs)
        
        # Compare vs fraction of equity
        if rule_frac:
            fold_comparison['vs_fraction_of_equity'] = compare_fold_metrics(rl_metrics, rule_frac)
        
        report['per_fold'].append(fold_comparison)
    
    # Aggregate verdicts
    verdicts_fs = [f['vs_fixed_shares']['verdict'] for f in report['per_fold'] 
                   if 'vs_fixed_shares' in f and 'verdict' in f['vs_fixed_shares']]
    verdicts_frac = [f['vs_fraction_of_equity']['verdict'] for f in report['per_fold']
                     if 'vs_fraction_of_equity' in f and 'verdict' in f['vs_fraction_of_equity']]
    
    report['aggregate'] = {
        'total_folds': len(report['per_fold']),
        'vs_fixed_shares': {
            'pass_count': verdicts_fs.count('PASS'),
            'marginal_count': verdicts_fs.count('MARGINAL'),
            'fail_count': verdicts_fs.count('FAIL'),
            'overall_verdict': 'PASS' if verdicts_fs.count('PASS') > len(verdicts_fs) / 2 else 
                              'MARGINAL' if verdicts_fs.count('FAIL') == 0 else 'FAIL'
        },
        'vs_fraction_of_equity': {
            'pass_count': verdicts_frac.count('PASS'),
            'marginal_count': verdicts_frac.count('MARGINAL'),
            'fail_count': verdicts_frac.count('FAIL'),
            'overall_verdict': 'PASS' if verdicts_frac.count('PASS') > len(verdicts_frac) / 2 else
                              'MARGINAL' if verdicts_frac.count('FAIL') == 0 else 'FAIL'
        }
    }
    
    # Summary conclusion
    overall_fs = report['aggregate']['vs_fixed_shares']['overall_verdict']
    overall_frac = report['aggregate']['vs_fraction_of_equity']['overall_verdict']
    
    if overall_fs == 'FAIL' or overall_frac == 'FAIL':
        report['final_conclusion'] = "RL FAILS to justify itself against rule baseline"
    elif overall_fs == 'MARGINAL' or overall_frac == 'MARGINAL':
        report['final_conclusion'] = "RL shows MARGINAL improvement over rule baseline"
    else:
        report['final_conclusion'] = "RL PASSES - demonstrates clear value over rule baseline"
    
    return report


def format_comparison_console(report: Dict[str, Any]) -> str:
    """Format comparison report for console output."""
    lines = []
    lines.append("=" * 80)
    lines.append("RL vs RULE BASELINE COMPARISON")
    lines.append("=" * 80)
    lines.append("")
    
    for fold_comp in report['per_fold']:
        fold_num = fold_comp['fold']
        lines.append(f"FOLD {fold_num}")
        lines.append("-" * 40)
        
        # RL metrics
        rl = fold_comp['rl_metrics']
        lines.append(f"  RL Results:")
        lines.append(f"    Total PnL: ${rl['total_test_pnl']:,.2f}" if rl['total_test_pnl'] else "    Total PnL: N/A")
        lines.append(f"    Win Rate: {rl['win_rate']*100:.1f}%" if rl['win_rate'] else "    Win Rate: N/A")
        lines.append(f"    Trades: {rl['total_trades']}")
        
        # vs Fixed Shares
        if 'vs_fixed_shares' in fold_comp:
            vs = fold_comp['vs_fixed_shares']
            lines.append(f"  vs Rule (Fixed Shares):")
            if 'error' not in vs:
                lines.append(f"    Rule PnL: ${vs['rule_total_pnl']:,.2f}")
                lines.append(f"    Difference: ${vs['pnl_difference']:,.2f}")
                lines.append(f"    Improvement: {vs['pnl_improvement_pct']:+.1f}%")
                lines.append(f"    VERDICT: {vs['verdict']}")
            else:
                lines.append(f"    Error: {vs['error']}")
        
        # vs Fraction of Equity
        if 'vs_fraction_of_equity' in fold_comp:
            vs = fold_comp['vs_fraction_of_equity']
            lines.append(f"  vs Rule (Fraction of Equity):")
            if 'error' not in vs:
                lines.append(f"    Rule PnL: ${vs['rule_total_pnl']:,.2f}")
                lines.append(f"    Improvement: {vs['pnl_improvement_pct']:+.1f}%")
                lines.append(f"    VERDICT: {vs['verdict']}")
        
        lines.append("")
    
    # Aggregate
    agg = report['aggregate']
    lines.append("=" * 80)
    lines.append("AGGREGATE RESULTS")
    lines.append("=" * 80)
    
    fs = agg['vs_fixed_shares']
    lines.append(f"vs Fixed Shares:")
    lines.append(f"  PASS: {fs['pass_count']}, MARGINAL: {fs['marginal_count']}, FAIL: {fs['fail_count']}")
    lines.append(f"  Overall: {fs['overall_verdict']}")
    
    frac = agg['vs_fraction_of_equity']
    lines.append(f"vs Fraction of Equity:")
    lines.append(f"  PASS: {frac['pass_count']}, MARGINAL: {frac['marginal_count']}, FAIL: {frac['fail_count']}")
    lines.append(f"  Overall: {frac['overall_verdict']}")
    
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"FINAL CONCLUSION: {report['final_conclusion']}")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def run_statistical_benchmarks(
    fold_data: Dict[str, Any],
    rl_episode_pnls: List[float],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run all statistical benchmarks for a single fold.

    Runs NaiveShort and Random agents on the same test setups,
    computes bootstrap CI on RL PnLs, and runs permutation tests.
    """
    # Build test setups from RL results (supports both quick-test and full WFO JSON shapes)
    test_setups = _extract_test_setups(fold_data)

    if not test_setups:
        return {'error': 'No test setups found'}

    test_start = fold_data.get('test_start', '')
    test_end = fold_data.get('test_end', '')
    env_config = {
        "initial_capital": 100000.0,
        "date_range": (test_start[:10], test_end[:10]) if test_start and test_end else None,
        "mode": "eval"
    }

    results = {}

    # Naive Short baseline
    if verbose:
        print("  Running Naive Short baseline...")
    naive_agent = NaiveShortAgent(entry_threshold=20.0)
    naive_metrics = evaluate_baseline_on_fold(
        agent=naive_agent, test_setups=test_setups,
        env_config=env_config, verbose=False
    )
    naive_pnls = [ep['pnl'] for ep in naive_metrics.get('per_episode_results', [])]
    results['naive_short'] = naive_metrics

    # Random agent (average of 5 seeds for stability)
    if verbose:
        print("  Running Random agent (5 seeds)...")
    random_pnls_all = []
    for seed in range(5):
        random_agent = RandomAgent(seed=seed * 1000)
        random_metrics = evaluate_baseline_on_fold(
            agent=random_agent, test_setups=test_setups,
            env_config=env_config, verbose=False
        )
        random_pnls_all.append(
            [ep['pnl'] for ep in random_metrics.get('per_episode_results', [])]
        )
    # Average across seeds per episode
    if random_pnls_all and random_pnls_all[0]:
        avg_random_pnls = [
            float(np.mean([seed_pnls[i] for seed_pnls in random_pnls_all
                           if i < len(seed_pnls)]))
            for i in range(len(random_pnls_all[0]))
        ]
    else:
        avg_random_pnls = []
    results['random_avg_pnls'] = avg_random_pnls
    results['random_total_pnl'] = sum(avg_random_pnls)

    # Bootstrap CI on RL PnLs
    if verbose:
        print("  Computing bootstrap confidence interval...")
    results['rl_bootstrap_ci'] = bootstrap_confidence_interval(rl_episode_pnls)

    # Permutation tests (RL vs each baseline)
    results['permutation_tests'] = {}
    if naive_pnls and len(naive_pnls) == len(rl_episode_pnls):
        results['permutation_tests']['naive_short'] = permutation_test(
            rl_episode_pnls, naive_pnls
        )
    if avg_random_pnls and len(avg_random_pnls) == len(rl_episode_pnls):
        results['permutation_tests']['random'] = permutation_test(
            rl_episode_pnls, avg_random_pnls
        )

    # Format report
    baseline_pnl_map = {}
    if naive_pnls:
        baseline_pnl_map['Naive Short'] = naive_pnls
    if avg_random_pnls:
        baseline_pnl_map['Random (5-seed avg)'] = avg_random_pnls

    perm_map = {}
    for key, ptest in results['permutation_tests'].items():
        display_name = 'Naive Short' if key == 'naive_short' else 'Random'
        perm_map[display_name] = ptest

    results['report'] = format_benchmark_report(
        rl_episode_pnls, baseline_pnl_map, perm_map
    )

    return results


def _build_headline_summary(
    rl_results_primary: Dict[str, Any],
    rl_results_all_seeds: List[Dict[str, Any]],
    rule_metrics: Dict[str, Any],
    sizing_mode: str,
) -> Dict[str, Any]:
    """
    Build the top-level ``rl`` / ``rule_baseline`` / ``delta`` summary that
    Phase 0 of the RL tuning plan requires. The rule baseline is deterministic
    on a given setup list, so it's run ONCE; the RL number is the mean across
    however many seed JSONs were passed in (typical: 3-seed mean).

    Args:
        rl_results_primary: One RL results JSON (used for setup list, dates,
            and the per-seed RL total reported as ``rl.total_test_pnl_seed1``).
        rl_results_all_seeds: List of all RL results JSONs (length 1 if only
            one seed was passed). Used to compute the multi-seed mean.
        rule_metrics: Rule-baseline metrics dict from ``evaluate_baseline_on_fold``.
        sizing_mode: Which rule-baseline sizing flavor this summary represents
            ("fixed_shares" or "fixed_fraction_of_equity").

    Returns:
        A dict with the headline fields: ``rl.total_test_pnl``,
        ``rule_baseline.total_test_pnl``, ``delta`` (rl - rule), per-setup PnL
        lists for both, and winner/loser stats.
    """
    rl_folds_primary = rl_results_primary.get('per_fold_results',
                                              rl_results_primary.get('folds', []))
    primary_fold = rl_folds_primary[0] if rl_folds_primary else {}
    primary_per_ep = (
        primary_fold.get('per_episode_results')
        or primary_fold.get('test_metrics', {}).get('per_episode_results')
        or []
    )

    # Aggregate RL across seeds (per-setup mean, then sum). Each seed's
    # quick_test_results.json holds one fold with the same 14 setups.
    rl_total_per_seed: List[float] = []
    rl_per_setup_pnls_per_seed: List[List[float]] = []
    rl_win_rate_per_seed: List[float] = []
    for seed_results in rl_results_all_seeds:
        folds = seed_results.get('per_fold_results', seed_results.get('folds', []))
        if not folds:
            continue
        f0 = folds[0]
        per_ep = (
            f0.get('per_episode_results')
            or f0.get('test_metrics', {}).get('per_episode_results')
            or []
        )
        pnls = [ep['pnl'] for ep in per_ep]
        rl_per_setup_pnls_per_seed.append(pnls)
        rl_total_per_seed.append(float(sum(pnls)))
        wins = sum(1 for p in pnls if p > 0)
        rl_win_rate_per_seed.append(wins / len(pnls) if pnls else 0.0)

    rl_mean_total = float(np.mean(rl_total_per_seed)) if rl_total_per_seed else None
    rl_std_total = float(np.std(rl_total_per_seed)) if len(rl_total_per_seed) > 1 else 0.0
    rl_mean_win_rate = float(np.mean(rl_win_rate_per_seed)) if rl_win_rate_per_seed else None

    # Rule baseline (single deterministic run on the same setups)
    rule_per_setup = rule_metrics.get('per_episode_results', [])
    rule_pnls = [ep['pnl'] for ep in rule_per_setup]
    rule_wl = _winner_loser_stats(rule_pnls)
    rule_total = rule_metrics.get('total_test_pnl')

    # Per-setup deltas: rule_pnl - mean_rl_pnl_for_that_setup
    rl_mean_per_setup: List[Optional[float]] = []
    if rl_per_setup_pnls_per_seed:
        n_setups = len(rl_per_setup_pnls_per_seed[0])
        for i in range(n_setups):
            vals = [seed_pnls[i] for seed_pnls in rl_per_setup_pnls_per_seed
                    if i < len(seed_pnls)]
            rl_mean_per_setup.append(float(np.mean(vals)) if vals else None)

    delta_total = (rule_total - rl_mean_total) if (rule_total is not None and rl_mean_total is not None) else None

    return {
        'sizing_mode': sizing_mode,
        'n_setups': len(primary_per_ep),
        'test_window': {
            'start': primary_fold.get('test_start', ''),
            'end': primary_fold.get('test_end', ''),
        },
        'setup_list': [{'symbol': ep['symbol'], 'date': ep['date']}
                       for ep in primary_per_ep],
        'rl': {
            'n_seeds': len(rl_results_all_seeds),
            'total_test_pnl': rl_mean_total,
            'total_test_pnl_per_seed': rl_total_per_seed,
            'total_test_pnl_std_across_seeds': rl_std_total,
            'win_rate': rl_mean_win_rate,
            'win_rate_per_seed': rl_win_rate_per_seed,
            'mean_per_setup_pnls': rl_mean_per_setup,
        },
        'rule_baseline': {
            'total_test_pnl': rule_total,
            'mean_episode_pnl': rule_metrics.get('mean_episode_pnl'),
            'win_rate': rule_metrics.get('win_rate'),
            'total_trades': rule_metrics.get('total_trades'),
            'mean_trades_per_episode': rule_metrics.get('mean_trades_per_episode'),
            'mean_winner': rule_wl['mean_winner'],
            'mean_loser': rule_wl['mean_loser'],
            'n_winners': rule_wl['n_winners'],
            'n_losers': rule_wl['n_losers'],
            'n_zero_pnl': rule_wl['n_zero_pnl'],
            'per_setup_pnls': rule_pnls,
            'per_episode_results': rule_per_setup,
        },
        # delta convention: rule - rl (so positive => rule beats rl)
        'delta': {
            'rule_minus_rl_total_pnl': delta_total,
            'rule_beats_rl': (delta_total is not None and delta_total > 0),
            'rl_minus_rule_total_pnl': -delta_total if delta_total is not None else None,
        },
    }


def main():
    parser = argparse.ArgumentParser(description='Compare RL vs Rule Baseline')
    parser.add_argument('--rl-results', type=str, required=True,
                        help='Path to RL results JSON from train_wfo.py. May be '
                             'a comma-separated list of paths to average across '
                             'seeds (rule baseline is deterministic, so it runs '
                             'once on the shared setup list).')
    parser.add_argument('--output', type=str, default='reports/rl_vs_rule_comparison.json',
                        help='Output path for comparison report')
    parser.add_argument('--run-baseline', action='store_true',
                        help='Actually run baseline evaluation (requires data)')
    parser.add_argument('--mock', action='store_true',
                        help='Generate mock comparison for testing')

    args = parser.parse_args()

    # Support multi-seed averaging: --rl-results can be a comma-separated list.
    rl_result_paths = [p.strip() for p in args.rl_results.split(',') if p.strip()]
    rl_results_all_seeds = [load_rl_results(p) for p in rl_result_paths]
    # Use the first one as "primary" for setup-list / dates / per-fold loop
    rl_results = rl_results_all_seeds[0]

    if args.mock:
        # Generate mock comparison for demonstration
        rule_fixed = [
            {'episodes_evaluated': 10, 'total_test_pnl': 8000.0, 'win_rate': 0.55,
             'mean_episode_pnl': 800.0, 'total_trades': 20, 'mean_trades_per_episode': 2.0,
             'per_episode_results': []}
        ]
        rule_frac = [
            {'episodes_evaluated': 10, 'total_test_pnl': 9000.0, 'win_rate': 0.58,
             'mean_episode_pnl': 900.0, 'total_trades': 18, 'mean_trades_per_episode': 1.8,
             'per_episode_results': []}
        ]
    elif args.run_baseline:
        # Actually run baseline evaluation
        print("Running rule baseline evaluations...")
        print(f"  RL seed JSONs supplied: {len(rl_result_paths)}")
        for p in rl_result_paths:
            print(f"    - {p}")
        rl_folds = rl_results.get('per_fold_results', rl_results.get('folds', []))

        rule_fixed = []
        rule_frac = []

        for fold in rl_folds:
            print(f"\nFold {fold.get('fold', '?')}:")

            # Fixed shares
            print("  Running fixed shares baseline...")
            result_fs = run_rule_baseline_on_fold(fold, "fixed_shares", verbose=True)
            rule_fixed.append(result_fs)

            # Fraction of equity
            print("  Running fraction of equity baseline...")
            result_frac = run_rule_baseline_on_fold(fold, "fixed_fraction_of_equity", verbose=True)
            rule_frac.append(result_frac)
    else:
        print("Error: Must specify --run-baseline to evaluate or --mock for demo")
        return

    # Generate report
    report = generate_comparison_report(rl_results, rule_fixed, rule_frac)

    # Headline summary (Phase 0 plan: top-level rl/rule_baseline/delta on a
    # single fold). Use fold-0 numbers since quick-test results only have one
    # fold; for multi-fold full WFO this would need extension.
    if rule_fixed and rule_fixed[0].get('episodes_evaluated', 0) > 0:
        report['summary_fixed_shares'] = _build_headline_summary(
            rl_results, rl_results_all_seeds, rule_fixed[0], "fixed_shares"
        )
    if rule_frac and rule_frac[0].get('episodes_evaluated', 0) > 0:
        report['summary_fraction_of_equity'] = _build_headline_summary(
            rl_results, rl_results_all_seeds, rule_frac[0], "fixed_fraction_of_equity"
        )
    report['rl_results_paths'] = rl_result_paths

    # Print to console
    print(format_comparison_console(report))

    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
