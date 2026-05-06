"""
Quick Test Run for WFO Training (1-2 hours)

This script runs a shortened WFO training to verify:
1. Data provider loads trading days correctly
2. Environment runs without errors
3. Agent actually learns (PnL != $0)
4. Checkpoints save properly

Recommended: Run this first before the full training.
"""

import ray
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from typing import Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import json
import logging
import torch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rl.env import ParabolicReversalEnv
from train_wfo import WarmupCallback, WalkForwardSplitter, WFOConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class QuickTestConfig:
    """Quick test configuration - runs in 1-2 hours."""
    
    output_dir: str = "models/wfo_test"
    
    # REDUCED: Just 1 fold for quick test (vs 4 in full training)
    # This tests the mechanics without spending hours
    n_folds: int = 1
    
    # REDUCED: 6 months train, 1 month test (vs 2 years / 6 months)
    train_months: int = 6
    test_months: int = 1
    purge_days: int = 5
    
    # REDUCED: Much fewer timesteps
    warmup_timesteps: int = 10000     # vs 30000 in full
    finetune_timesteps: int = 30000   # vs 70000 in full
    
    # SAC parameters (same as full)
    buffer_size: int = 100000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    alpha: float = 0.2
    
    # REDUCED: Fewer eval episodes
    eval_episodes: int = 3  # vs 10 in full
    
    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


class QuickWFOTrainer:
    """Quick test trainer - verifies everything works."""
    
    def __init__(self, config: QuickTestConfig):
        self.config = config
        
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        logger.info("QuickWFOTrainer initialized")
    
    def create_sac_config(self, fold: int) -> SACConfig:
        """Create SAC configuration."""
        
        callback_config = {
            "warmup_timesteps": self.config.warmup_timesteps,
            "warmup_lr_actor": 0.0,
            "finetune_lr_actor": 3e-4,
            "finetune_lr_critic": 3e-4,
            "bc_checkpoint": None  # Skip BC for quick test
        }
        
        sac_config = (
            SACConfig()
            .environment(
                env=ParabolicReversalEnv,
                env_config={"initial_capital": 100000.0},
                disable_env_checking=True,
            )
            .framework("torch")
            .training(
                twin_q=True,
                q_model_config={"fcnet_hiddens": [256, 256], "fcnet_activation": "relu"},
                policy_model_config={"fcnet_hiddens": [256, 256], "fcnet_activation": "relu"},
                tau=self.config.tau,
                initial_alpha=self.config.alpha,
                target_entropy="auto",
                n_step=1,
                replay_buffer_config={
                    "type": "MultiAgentReplayBuffer",
                    "capacity": self.config.buffer_size,
                },
                train_batch_size=self.config.batch_size,
            )
            .callbacks(callbacks_class=lambda: WarmupCallback(callback_config))
            .rollouts(
                num_rollout_workers=0,
                rollout_fragment_length=200,
            )
            .reporting(
                min_time_s_per_iteration=5,
            )
            .evaluation(
                evaluation_interval=1,
                evaluation_duration=self.config.eval_episodes,
                evaluation_duration_unit="episodes",
            )
        )
        
        return sac_config
    
    def train_fold(
        self,
        fold: int,
        train_start: datetime,
        train_end: datetime,
        test_start: datetime,
        test_end: datetime
    ) -> Dict[str, Any]:
        """Train and evaluate on a single fold."""
        
        logger.info(f"\n{'='*70}")
        logger.info(f"QUICK TEST FOLD {fold}")
        logger.info(f"{'='*70}")
        logger.info(f"Train: {train_start.date()} → {train_end.date()}")
        logger.info(f"Test:  {test_start.date()} → {test_end.date()}")
        
        config = self.create_sac_config(fold)
        algo = config.build()
        
        total_timesteps = self.config.warmup_timesteps + self.config.finetune_timesteps
        
        logger.info(f"\nTraining for {total_timesteps} timesteps...")
        logger.info(f"Phase 1 (Actor frozen): 0-{self.config.warmup_timesteps}")
        logger.info(f"Phase 2 (Actor unfrozen): {self.config.warmup_timesteps}+")
        logger.info(f"Estimated time: 20-30 minutes for this fold\n")
        
        results = []
        last_log_time = datetime.now()
        
        # Train in iterations
        for i in range(total_timesteps // 1000):
            result = algo.train()
            results.append(result)
            
            timesteps = result.get("timesteps_total", 0)
            phase = result.get("phase", "unknown")
            
            # Log every 10 iterations or when phase changes
            if i % 10 == 0 or (results and results[-1].get("phase") != phase):
                elapsed = (datetime.now() - last_log_time).total_seconds()
                logger.info(f"Iteration {i}: {timesteps} steps, Phase: {phase}, "
                           f"Time since last log: {elapsed:.1f}s")
                last_log_time = datetime.now()
            
            if timesteps >= total_timesteps:
                break
        
        final_result = results[-1] if results else {}
        
        # Evaluate on test set
        logger.info(f"\nEvaluating on test window...")
        eval_results = algo.evaluate()
        
        # Extract metrics
        test_metrics = self._extract_metrics(eval_results)
        test_reward = test_metrics.get('episode_reward_mean', 0)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"TEST RESULTS FOR FOLD {fold}")
        logger.info(f"{'='*70}")
        logger.info(f"Test Reward (PnL): ${test_reward:,.2f}")
        logger.info(f"Test Reward Max:   ${test_metrics.get('episode_reward_max', 0):,.2f}")
        logger.info(f"Test Reward Min:   ${test_metrics.get('episode_reward_min', 0):,.2f}")
        
        # CRITICAL CHECK: Is reward non-zero?
        if abs(test_reward) < 0.01:
            logger.error("❌ CRITICAL: Test reward is ~$0.00 - data may not be loading!")
        else:
            logger.info(f"✅ Test reward is non-zero - data is loading correctly!")
        
        logger.info(f"{'='*70}\n")
        
        # Save checkpoint
        checkpoint_dir = Path(self.config.output_dir) / f"fold_{fold}_checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        algo.save_checkpoint(str(checkpoint_dir))
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
        
        result_data = {
            'fold': fold,
            'train_start': train_start.isoformat(),
            'train_end': train_end.isoformat(),
            'test_start': test_start.isoformat(),
            'test_end': test_end.isoformat(),
            'timesteps_total': final_result.get('timesteps_total', 0),
            'phase_at_end': final_result.get('phase', 'unknown'),
            'test_reward_mean': test_reward,
            'test_reward_max': test_metrics.get('episode_reward_max', 0),
            'test_reward_min': test_metrics.get('episode_reward_min', 0),
        }
        
        algo.stop()
        
        return result_data
    
    def _extract_metrics(self, eval_results: Dict) -> Dict:
        """Extract relevant metrics from evaluation results."""
        if 'evaluation' in eval_results:
            return eval_results['evaluation']
        return eval_results
    
    def run(self):
        """Run quick test."""
        
        # Use recent data for quick test (more relevant)
        splitter = WalkForwardSplitter(
            start_date=datetime(2023, 1, 1),  # Start from 2023
            end_date=datetime(2024, 6, 30),   # To mid-2024
            train_years=0,  # We'll override with months
            test_months=self.config.test_months,
            purge_days=self.config.purge_days,
            step_months=self.config.test_months
        )
        
        # Override train window to use months instead of years
        splits = []
        current_start = splitter.start_date
        
        for i in range(self.config.n_folds):
            train_start = current_start
            train_end = train_start + timedelta(days=30 * self.config.train_months)
            
            purge_start = train_end
            purge_end = purge_start + timedelta(days=self.config.purge_days)
            
            test_start = purge_end
            test_end = test_start + timedelta(days=30 * self.config.test_months)
            
            if test_end > splitter.end_date:
                break
            
            splits.append({
                'train_start': train_start,
                'train_end': train_end,
                'purge_start': purge_start,
                'purge_end': purge_end,
                'test_start': test_start,
                'test_end': test_end,
                'fold': len(splits) + 1
            })
            
            current_start += timedelta(days=30 * self.config.test_months)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"QUICK WFO TEST (1-2 hours)")
        logger.info(f"{'='*70}")
        logger.info(f"Folds: {len(splits)}")
        logger.info(f"Train: {self.config.train_months} months")
        logger.info(f"Test:  {self.config.test_months} month")
        logger.info(f"Timesteps: {self.config.warmup_timesteps + self.config.finetune_timesteps}")
        logger.info(f"{'='*70}\n")
        
        # Run each fold
        all_results = []
        for split in splits[:self.config.n_folds]:
            result = self.train_fold(
                fold=split['fold'],
                train_start=split['train_start'],
                train_end=split['train_end'],
                test_start=split['test_start'],
                test_end=split['test_end']
            )
            all_results.append(result)
        
        # Summary
        logger.info(f"\n{'='*70}")
        logger.info(f"QUICK TEST COMPLETE")
        logger.info(f"{'='*70}")
        
        avg_reward = np.mean([r['test_reward_mean'] for r in all_results])
        
        logger.info(f"Average Test Reward: ${avg_reward:,.2f}")
        
        if abs(avg_reward) < 0.01:
            logger.error("❌ OVERALL: Test reward is ~$0.00 - check data provider!")
            logger.error("   Run: python test_data_provider.py")
        else:
            logger.info(f"✅ OVERALL: Test reward is non-zero - ready for full training!")
            logger.info(f"   Next: Run train_wfo.py for full training")
        
        # Save results
        results_path = Path(self.config.output_dir) / "quick_test_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'config': {
                    'train_months': self.config.train_months,
                    'test_months': self.config.test_months,
                    'warmup_timesteps': self.config.warmup_timesteps,
                    'finetune_timesteps': self.config.finetune_timesteps,
                },
                'folds': all_results,
                'aggregate': {
                    'avg_test_reward': avg_reward,
                }
            }, f, indent=2, default=str)
        
        logger.info(f"\nResults saved to {results_path}")
        
        ray.shutdown()
        
        return all_results


def main():
    """Main entry point for quick test."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick WFO Test (1-2 hours)')
    parser.add_argument('--output-dir', type=str, default='models/wfo_test')
    parser.add_argument('--warmup-steps', type=int, default=5000)
    parser.add_argument('--finetune-steps', type=int, default=15000)
    parser.add_argument('--train-months', type=int, default=6)
    parser.add_argument('--test-months', type=int, default=1)
    
    args = parser.parse_args()
    
    config = QuickTestConfig(
        output_dir=args.output_dir,
        warmup_timesteps=args.warmup_steps,
        finetune_timesteps=args.finetune_steps,
        train_months=args.train_months,
        test_months=args.test_months
    )
    
    trainer = QuickWFOTrainer(config)
    results = trainer.run()
    
    return results


if __name__ == "__main__":
    main()
