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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.baselines.rule_baseline import (
    create_v5_relaxed_agent_fixed_shares,
    create_v5_relaxed_agent_fraction_of_equity
)
from src.baselines.evaluate_baseline import evaluate_baseline_on_fold
from src.utils.metrics import compute_fold_metrics


def load_rl_results(path: str) -> Dict[str, Any]:
    """Load RL results from train_wfo.py output."""
    with open(path, 'r') as f:
        return json.load(f)


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
    
    # Build test setups from RL results
    # For now, we reconstruct from per_episode_results if available
    test_setups = []
    if 'test_metrics' in fold_data and 'per_episode_results' in fold_data['test_metrics']:
        for ep in fold_data['test_metrics']['per_episode_results']:
            test_setups.append({
                'symbol': ep['symbol'],
                'date': ep['date']
            })
    
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


def main():
    parser = argparse.ArgumentParser(description='Compare RL vs Rule Baseline')
    parser.add_argument('--rl-results', type=str, required=True,
                        help='Path to RL results JSON from train_wfo.py')
    parser.add_argument('--output', type=str, default='reports/rl_vs_rule_comparison.json',
                        help='Output path for comparison report')
    parser.add_argument('--run-baseline', action='store_true',
                        help='Actually run baseline evaluation (requires data)')
    parser.add_argument('--mock', action='store_true',
                        help='Generate mock comparison for testing')
    
    args = parser.parse_args()
    
    # Load RL results
    rl_results = load_rl_results(args.rl_results)
    
    if args.mock:
        # Generate mock comparison for demonstration
        rule_fixed = [
            {'episodes_evaluated': 10, 'total_test_pnl': 8000.0, 'win_rate': 0.55,
             'mean_episode_pnl': 800.0, 'total_trades': 20, 'mean_trades_per_episode': 2.0}
        ]
        rule_frac = [
            {'episodes_evaluated': 10, 'total_test_pnl': 9000.0, 'win_rate': 0.58,
             'mean_episode_pnl': 900.0, 'total_trades': 18, 'mean_trades_per_episode': 1.8}
        ]
    elif args.run_baseline:
        # Actually run baseline evaluation
        print("Running rule baseline evaluations...")
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
