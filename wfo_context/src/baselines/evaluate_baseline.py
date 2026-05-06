"""
Unified Evaluation Harness for Baseline Agents

This module provides standardized evaluation that runs ANY baseline agent
through the same test protocol as RL:
- Same test setups
- Same environment (execution/accounting/costs)
- Same metrics computation

This ensures fair comparison between RL and baselines.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Type
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rl.env import ParabolicReversalEnv
from src.utils.metrics import compute_fold_metrics


def evaluate_baseline_on_fold(
    agent,
    test_setups: List[Dict[str, Any]],
    env_config: Optional[Dict] = None,
    max_steps: int = 500,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Evaluate a baseline agent on the same test setups as RL.
    
    This is the SINGLE EVALUATION PATH for all baselines.
    Uses the SAME environment and accounting as RL training.
    
    Args:
        agent: Baseline agent with act(obs, info) method
        test_setups: List of {'symbol': str, 'date': str} dicts
        env_config: Environment configuration (same as RL eval)
        max_steps: Max steps per episode
        verbose: Print progress
        
    Returns:
        Dict with standardized metrics (same keys as train_wfo.py)
    """
    if env_config is None:
        env_config = {
            "initial_capital": 100000.0,
            "mode": "eval"
        }
    
    episodes_requested = len(test_setups)
    episodes_failed = 0
    per_episode_results = []
    
    if verbose:
        print(f"Evaluating on {episodes_requested} test setups...")
    
    for episode_idx, setup in enumerate(test_setups, 1):
        symbol = setup['symbol']
        date_str = setup['date']
        
        try:
            # Create fresh environment
            env = ParabolicReversalEnv(config=env_config)
            
            # Reset with fixed setup (deterministic)
            obs, info = env.reset(options={
                "fixed_setup": {"symbol": symbol, "date": date_str}
            })
            
            # Verify load succeeded
            if env.data_provider.current_symbol is None:
                if verbose:
                    print(f"  [{episode_idx}/{episodes_requested}] FAILED: {symbol} {date_str}")
                episodes_failed += 1
                continue
            
            # Run episode
            agent.reset()  # Reset agent state for new episode
            done = False
            truncated = False
            step_count = 0
            
            while not (done or truncated) and step_count < max_steps:
                obs_dict = obs if isinstance(obs, dict) else {'state': obs}
                
                # Get action from baseline agent
                action = agent.act(obs, info)
                
                # None means hold current position
                if action is None:
                    action = np.array([env.current_position / 100.0])  # Normalize
                
                # Step environment
                obs, reward, done, truncated, info = env.step(action)
                step_count += 1
            
            # Record result
            per_episode_results.append({
                'symbol': symbol,
                'date': date_str,
                'pnl': env.episode_pnl,
                'trades': env.episode_trades
            })
            
            if verbose and (episode_idx % 10 == 0 or episode_idx <= 5):
                print(f"  [{episode_idx}/{episodes_requested}] {symbol} {date_str} | "
                      f"PnL: ${env.episode_pnl:,.2f} | Trades: {env.episode_trades}")
                
        except Exception as e:
            if verbose:
                print(f"  [{episode_idx}/{episodes_requested}] EXCEPTION: {symbol} {date_str} - {e}")
            episodes_failed += 1
            continue
    
    # Compute standardized metrics
    metrics = compute_fold_metrics(
        per_episode_results=per_episode_results,
        episodes_requested=episodes_requested,
        episodes_failed=episodes_failed
    )
    
    if verbose:
        print(f"\nEvaluation complete:")
        print(f"  Evaluated: {metrics['episodes_evaluated']}/{episodes_requested}")
        print(f"  Failed: {episodes_failed}")
        if metrics['episodes_evaluated'] > 0:
            print(f"  Total PnL: ${metrics['total_test_pnl']:,.2f}")
            print(f"  Mean PnL: ${metrics['mean_episode_pnl']:,.2f}")
            print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
    
    return metrics


def compare_to_rule_baseline(
    rl_metrics: Dict[str, Any],
    rule_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare RL metrics to rule baseline.
    
    Returns comparison dict with percentage improvements.
    """
    if rl_metrics['episodes_evaluated'] == 0 or rule_metrics['episodes_evaluated'] == 0:
        return {'error': 'Insufficient data for comparison'}
    
    rl_pnl = rl_metrics['total_test_pnl']
    rule_pnl = rule_metrics['total_test_pnl']
    
    if rule_pnl != 0:
        pnl_improvement_pct = ((rl_pnl - rule_pnl) / abs(rule_pnl)) * 100
    else:
        pnl_improvement_pct = float('inf') if rl_pnl > 0 else float('-inf')
    
    return {
        'rl_total_pnl': rl_pnl,
        'rule_total_pnl': rule_pnl,
        'pnl_difference': rl_pnl - rule_pnl,
        'pnl_improvement_pct': pnl_improvement_pct,
        'rl_win_rate': rl_metrics['win_rate'],
        'rule_win_rate': rule_metrics['win_rate'],
        'rl_avg_trades': rl_metrics['mean_trades_per_episode'],
        'rule_avg_trades': rule_metrics['mean_trades_per_episode'],
        'verdict': 'PASS' if pnl_improvement_pct > 10 else 'MARGINAL' if pnl_improvement_pct > 0 else 'FAIL'
    }
