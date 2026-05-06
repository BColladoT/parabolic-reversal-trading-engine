"""
Shared metrics computation for RL and baselines.

This module provides standardized fold-level metrics computation
to ensure all methods (RL, rule-based, supervised) report identically.
"""

import numpy as np
from typing import List, Dict, Any, Optional


def compute_fold_metrics(
    per_episode_results: List[Dict[str, Any]],
    episodes_requested: int,
    episodes_failed: int = 0
) -> Dict[str, Any]:
    """
    Compute standardized fold metrics from episode results.
    
    This is the SINGLE SOURCE OF TRUTH for metrics computation.
    Both RL and baselines use this function to ensure comparable results.
    
    Args:
        per_episode_results: List of episode result dicts with 'pnl', 'trades', etc.
        episodes_requested: Number of episodes attempted (including failed)
        episodes_failed: Number of episodes that failed to load
        
    Returns:
        Dict with standardized metric names and values.
        Returns None for aggregate metrics if no episodes evaluated.
    """
    episodes_evaluated = len(per_episode_results)
    
    # Handle zero-evaluation case honestly
    if episodes_evaluated == 0:
        return {
            'episodes_requested': episodes_requested,
            'episodes_evaluated': 0,
            'episodes_failed_to_load': episodes_failed,
            'total_test_pnl': None,
            'mean_episode_pnl': None,
            'median_episode_pnl': None,
            'min_episode_pnl': None,
            'max_episode_pnl': None,
            'win_rate': None,
            'total_trades': 0,
            'mean_trades_per_episode': None,
            'per_episode_results': []
        }
    
    # Extract arrays for computation
    pnls = [e['pnl'] for e in per_episode_results]
    trades = [e.get('trades', 0) for e in per_episode_results]
    winning = sum(1 for p in pnls if p > 0)
    
    return {
        # Counts
        'episodes_requested': episodes_requested,
        'episodes_evaluated': episodes_evaluated,
        'episodes_failed_to_load': episodes_failed,
        
        # PnL metrics
        'total_test_pnl': float(sum(pnls)),
        'mean_episode_pnl': float(np.mean(pnls)),
        'median_episode_pnl': float(np.median(pnls)),
        'min_episode_pnl': float(min(pnls)),
        'max_episode_pnl': float(max(pnls)),
        
        # Win rate
        'win_rate': winning / episodes_evaluated,
        'winning_episodes': winning,
        'losing_episodes': episodes_evaluated - winning,
        
        # Trade metrics
        'total_trades': int(sum(trades)),
        'mean_trades_per_episode': float(np.mean(trades)),
        
        # Raw results for audit
        'per_episode_results': per_episode_results
    }


def format_metrics_for_display(metrics: Dict[str, Any]) -> str:
    """Format metrics dict for human-readable display."""
    if metrics['episodes_evaluated'] == 0:
        return "No valid evaluations"
    
    return (
        f"Episodes: {metrics['episodes_evaluated']}/{metrics['episodes_requested']} "
        f"({metrics['episodes_failed_to_load']} failed)\n"
        f"Total PnL: ${metrics['total_test_pnl']:,.2f}\n"
        f"Mean PnL: ${metrics['mean_episode_pnl']:,.2f}\n"
        f"Median PnL: ${metrics['median_episode_pnl']:,.2f}\n"
        f"Min/Max: ${metrics['min_episode_pnl']:,.2f} / ${metrics['max_episode_pnl']:,.2f}\n"
        f"Win Rate: {metrics['win_rate']*100:.1f}%\n"
        f"Total Trades: {metrics['total_trades']}\n"
        f"Trades/Episode: {metrics['mean_trades_per_episode']:.2f}"
    )
