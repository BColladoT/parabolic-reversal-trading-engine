"""
Walk-Forward Optimization (WFO) with Ray RLlib

This script implements WFO using Ray RLlib's native APIs with a custom
callback for two-phase training (Actor freezing/unfreezing).

Architecture:
- Uses Ray RLlib's SAC algorithm via algo.train()
- Custom WarmupCallback handles phase transitions
- Accesses policy via algo.get_policy()
- Dynamically controls Actor optimizer LR through callback hooks

Author: AI Agent
Date: 2026-03-12
"""

import ray
from ray import tune
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.policy import Policy
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.typing import PolicyID
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import polars as pl
import json
import logging
import torch
import sys

# Add project root to path (parent of src/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rl.env import ParabolicReversalEnv
from src.rl.agent import SACConfig as AgentSACConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class WFOConfig:
    """Configuration for Walk-Forward Optimization."""
    
    # Paths
    bc_checkpoint: str = "models/behavioral_cloning/bc_actor_rllib.pt"
    output_dir: str = "models/wfo"
    
    # Walk-Forward parameters
    train_years: int = 2
    test_months: int = 6
    purge_days: int = 10
    step_months: int = 6
    
    # Phase 1: Critic Warm-Up (Actor Frozen)
    warmup_timesteps: int = 30000
    warmup_lr_critic: float = 3e-4
    warmup_lr_actor: float = 0.0  # Actor frozen

    # Phase 2: SAC Fine-Tuning (Actor Unfrozen)
    finetune_timesteps: int = 70000
    finetune_lr_actor: float = 3e-4
    finetune_lr_critic: float = 3e-4
    
    # SAC parameters
    buffer_size: int = 1000000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    alpha: float = 0.2
    
    # Evaluation
    eval_episodes: int = 10
    
    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


# =============================================================================
# Ray RLlib Callback for Two-Phase Training
# =============================================================================

class WarmupCallback(DefaultCallbacks):
    """
    Custom callback for two-phase SAC training with Actor freezing.
    
    Phase 1 (Warm-Up): Actor LR = 0.0 (frozen), only Critics update
    Phase 2 (Fine-Tuning): Actor LR = 3e-4 (unfrozen), standard SAC
    
    This callback hooks into RLlib's training loop to dynamically control
    the Actor optimizer's learning rate based on timestep count.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        self.config = config or {}
        self.warmup_timesteps = self.config.get("warmup_timesteps", 20000)
        self.warmup_lr_actor = self.config.get("warmup_lr_actor", 0.0)
        self.finetune_lr_actor = self.config.get("finetune_lr_actor", 3e-4)
        self.finetune_lr_critic = self.config.get("finetune_lr_critic", 3e-4)
        self.bc_checkpoint = self.config.get("bc_checkpoint", None)
        
        self.phase = 1
        self.actor_frozen = False
        self.initialized = False
        
        logger.info(f"WarmupCallback initialized")
        logger.info(f"Phase 1 (Actor frozen): 0 to {self.warmup_timesteps} timesteps")
        logger.info(f"Phase 2 (Actor unfrozen): {self.warmup_timesteps}+ timesteps")
        logger.info(f"BC checkpoint: {self.bc_checkpoint}")
    
    def on_algorithm_init(self, *, algorithm: "Algorithm", **kwargs) -> None:
        """
        Called when algorithm initializes.
        Load BC weights and freeze Actor immediately.
        """
        logger.info("="*60)
        logger.info("ALGORITHM INIT - Loading BC weights and freezing Actor")
        logger.info("="*60)
        
        # Get the policy
        policy = algorithm.get_policy()
        
        if policy is None:
            logger.warning("Policy not available at init")
            return
        
        # Load BC weights if available
        if self.bc_checkpoint and Path(self.bc_checkpoint).exists():
            self._load_bc_weights(policy, self.bc_checkpoint)
            logger.info(f"Successfully loaded BC checkpoint: {self.bc_checkpoint}")
        else:
            logger.warning(f"BC checkpoint not found: {self.bc_checkpoint}")
        
        # Freeze Actor for Phase 1
        self._freeze_actor(policy)
        self.initialized = True
        
        logger.info("Actor frozen - ready for Phase 1 (Critic warm-up)")
    
    def _load_bc_weights(self, policy: Policy, checkpoint_path: str):
        """Load pre-trained Behavioral Cloning Actor weights."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
            # Get the model from policy
            model = policy.model
            
            # Load actor weights - need to match keys
            actor_state = checkpoint.get('model_state_dict', checkpoint.get('actor_state_dict', {}))
            
            if actor_state:
                # Try to load compatible weights
                model.load_state_dict(actor_state, strict=False)
                logger.info(f"Loaded BC weights from {checkpoint_path}")
            else:
                logger.warning("No actor state found in checkpoint")
                
        except Exception as e:
            logger.error(f"Failed to load BC weights: {e}")
    
    def _is_actor_optimizer(self, opt, model) -> bool:
        """
        Check if optimizer is the Actor optimizer by comparing parameters.
        
        Args:
            opt: PyTorch optimizer
            model: RLlib model
            
        Returns:
            True if this optimizer controls Actor parameters
        """
        # Get Actor parameters from model
        actor_params = set()
        if hasattr(model, 'action_model'):
            # RLlib SAC uses action_model for the policy network
            actor_params = set(id(p) for p in model.action_model.parameters())
        elif hasattr(model, 'policy'):
            actor_params = set(id(p) for p in model.policy.parameters())
        
        # Check if optimizer contains any Actor parameters
        for param_group in opt.param_groups:
            for param in param_group.get('params', []):
                if id(param) in actor_params:
                    return True
        return False
    
    def _freeze_actor(self, policy: Policy):
        """
        FREEZE Actor network with two-layer protection:
        1. requires_grad = False (hard graph freeze)
        2. LR = 0.0 (secondary protection)
        
        Args:
            policy: RLlib Policy object (TorchPolicy)
        """
        # EXPLICITLY extract model from policy
        model = policy.model
        
        # LAYER 1: Hard graph freeze - target Actor module directly
        # RLlib SAC uses action_model for the policy/actor network
        frozen_count = 0
        if hasattr(model, 'action_model'):
            for param in model.action_model.parameters():
                param.requires_grad = False
                frozen_count += 1
            logger.info(f"Froze {frozen_count} parameters in model.action_model")
        elif hasattr(model, 'policy'):
            for param in model.policy.parameters():
                param.requires_grad = False
                frozen_count += 1
            logger.info(f"Froze {frozen_count} parameters in model.policy")
        else:
            logger.warning("Could not find action_model or policy in model")
        
        # LAYER 2: Zero out Actor optimizer learning rate
        # EXPLICITLY extract optimizers - RLlib stores as list, don't call it
        optimizers = []
        if hasattr(policy, 'get_optimizers'):
            optimizers = policy.get_optimizers()
        elif hasattr(policy, '_optimizers'):
            optimizers = policy._optimizers
        
        if optimizers:
            for i, opt in enumerate(optimizers):
                # Identify Actor optimizer safely
                if self._is_actor_optimizer(opt, model):
                    # Set LR to 0 to freeze
                    for param_group in opt.param_groups:
                        param_group['lr'] = self.warmup_lr_actor
                    
                    logger.info(f"Actor optimizer {i} frozen:")
                    logger.info(f"  - requires_grad = False (hard freeze)")
                    logger.info(f"  - LR = {self.warmup_lr_actor}")
                    self.actor_frozen = True
                    break
        else:
            logger.warning("Could not access policy optimizers")
    
    def _unfreeze_actor(self, policy: Policy):
        """
        UNFREEZE Actor network with Adam momentum state flush:
        1. Restore requires_grad = True
        2. Clear optimizer state (momentum buffers) - ONLY for Actor
        3. Restore learning rate
        
        CRITICAL: Only clears Actor optimizer state, preserves Critic momentum!
        
        Args:
            policy: RLlib Policy object (TorchPolicy)
        """
        # EXPLICITLY extract model from policy
        model = policy.model
        
        # EXPLICITLY extract optimizers - RLlib stores as list, don't call it
        optimizers = []
        if hasattr(policy, 'get_optimizers'):
            optimizers = policy.get_optimizers()
        elif hasattr(policy, '_optimizers'):
            optimizers = policy._optimizers
        
        if optimizers:
            for i, opt in enumerate(optimizers):
                # SAFETY CHECK: Only modify Actor optimizer, leave Critics alone
                if self._is_actor_optimizer(opt, model):
                    # STEP 1: Clear Adam's internal state (momentum buffers)
                    # This prevents accumulated garbage momentum from exploding
                    if len(opt.state) > 0:
                        opt.state.clear()
                        logger.info(f"Cleared optimizer {i} state (momentum buffers flushed)")
                    
                    # STEP 2: Restore requires_grad = True (Actor only)
                    unfrozen_count = 0
                    if hasattr(model, 'action_model'):
                        for param in model.action_model.parameters():
                            param.requires_grad = True
                            unfrozen_count += 1
                    elif hasattr(model, 'policy'):
                        for param in model.policy.parameters():
                            param.requires_grad = True
                            unfrozen_count += 1
                    
                    # STEP 3: Restore learning rate
                    for param_group in opt.param_groups:
                        param_group['lr'] = self.finetune_lr_actor
                    
                    logger.info(f"Actor optimizer {i} unfrozen:")
                    logger.info(f"  - {unfrozen_count} parameters now require_grad = True")
                    logger.info(f"  - Momentum state CLEARED (Critics preserved)")
                    logger.info(f"  - LR = {self.finetune_lr_actor}")
                    
                    self.actor_frozen = False
                    break
        else:
            logger.warning("Could not access policy optimizers")
    
    def on_train_result(self, *, algorithm: "Algorithm", result: Dict, **kwargs) -> None:
        """
        Called after each training iteration.
        Check timestep count and transition from Phase 1 to Phase 2.
        """
        timesteps_total = result.get("timesteps_total", 0)
        
        # Phase transition check
        if self.phase == 1 and timesteps_total >= self.warmup_timesteps:
            logger.info("\n" + "="*60)
            logger.info(f"PHASE TRANSITION: Warm-Up → Fine-Tuning")
            logger.info(f"Timesteps: {timesteps_total}")
            logger.info("="*60 + "\n")
            
            self.phase = 2
            
            # Get policy and unfreeze actor
            policy = algorithm.get_policy()
            if policy:
                self._unfreeze_actor(policy)
            else:
                logger.warning("Could not get policy for unfreezing")
        
        # Log current phase
        if self.phase == 1:
            result["phase"] = "warmup_actor_frozen"
            result["actor_lr"] = self.warmup_lr_actor
        else:
            result["phase"] = "finetuning_actor_active"
            result["actor_lr"] = self.finetune_lr_actor
        
        result["phase_num"] = self.phase


# =============================================================================
# Walk-Forward Splitter
# =============================================================================

class WalkForwardSplitter:
    """Chronological train/test splitter with purge embargo."""
    
    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        train_years: int = 2,
        test_months: int = 6,
        purge_days: int = 10,
        step_months: int = 6
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.train_years = train_years
        self.test_months = test_months
        self.purge_days = purge_days
        self.step_months = step_months
        
        self.splits = self._generate_splits()
        logger.info(f"Generated {len(self.splits)} WFO splits")
    
    def _generate_splits(self) -> List[Dict[str, datetime]]:
        """Generate chronological train/test splits with embargo."""
        splits = []
        current_start = self.start_date
        
        while True:
            train_start = current_start
            train_end = train_start + timedelta(days=365 * self.train_years)
            
            purge_start = train_end
            purge_end = purge_start + timedelta(days=self.purge_days)
            
            test_start = purge_end
            test_end = test_start + timedelta(days=30 * self.test_months)
            
            if test_end > self.end_date:
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
            
            current_start += timedelta(days=30 * self.step_months)
        
        return splits
    
    def __len__(self) -> int:
        return len(self.splits)
    
    def __getitem__(self, idx: int) -> Dict[str, datetime]:
        return self.splits[idx]


# =============================================================================
# Ray RLlib Training with WFO
# =============================================================================

class WalkForwardRLlibTrainer:
    """Orchestrates WFO using Ray RLlib's native APIs."""
    
    def __init__(self, config: WFOConfig):
        self.config = config
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        logger.info("WalkForwardRLlibTrainer initialized")
    
    def create_sac_config(self, fold: int, train_start: datetime, train_end: datetime) -> SACConfig:
        """Create Ray RLlib SAC configuration with custom callback."""
        
        # Callback config
        callback_config = {
            "warmup_timesteps": self.config.warmup_timesteps,
            "warmup_lr_actor": self.config.warmup_lr_actor,
            "finetune_lr_actor": self.config.finetune_lr_actor,
            "finetune_lr_critic": self.config.finetune_lr_critic,
            "bc_checkpoint": self.config.bc_checkpoint
        }
        
        # CRITICAL: Date range filter to prevent WFO data leakage
        # Training environment MUST ONLY sample from training period
        train_date_range = (train_start.strftime('%Y-%m-%d'), train_end.strftime('%Y-%m-%d'))
        
        logger.info(f"Fold {fold} training date range: {train_date_range[0]} to {train_date_range[1]}")
        
        # Build SAC config (legacy API stack for stability)
        # 
        # ACTIVE PATH: Plain RLlib SAC without custom action masking.
        # The MaskedSAC model in agent.py is NOT wired into this trainer.
        # Action constraints are enforced by environment masking_penalty (-10.0).
        #
        # To use model-level action masking (EXPERIMENTAL):
        #   1. Import build_sac_config from agent.py
        #   2. Call build_sac_config(custom_model=True) - see warnings in agent.py
        #
        logger.info("[ACTIVE PATH] Using plain RLlib SAC (no custom model masking)")
        logger.info("Action constraints enforced by environment penalty")
        
        sac_config = (
            SACConfig()
            .environment(
                env=ParabolicReversalEnv,
                env_config={
                    "initial_capital": 100000.0,
                    "date_range": train_date_range,  # CRITICAL: Prevents data leakage
                    "seed": fold * 1000,  # Unique seed per fold for reproducibility
                },
                disable_env_checking=True,
            )
            .framework("torch")
            .training(
                twin_q=True,
                q_model_config={
                    "fcnet_hiddens": [256, 256],
                    "fcnet_activation": "relu",
                },
                policy_model_config={
                    "fcnet_hiddens": [256, 256],
                    "fcnet_activation": "relu",
                },
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
            .callbacks(
                callbacks_class=lambda: WarmupCallback(callback_config)
            )
            .rollouts(
                num_rollout_workers=0,
                rollout_fragment_length=200,
            )
            .reporting(
                min_time_s_per_iteration=5,
                keep_per_episode_custom_metrics=True,
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
        """Train and evaluate on a single WFO fold using Ray RLlib."""
        
        logger.info(f"\n{'='*70}")
        logger.info(f"WFO FOLD {fold}")
        logger.info(f"{'='*70}")
        logger.info(f"Train: {train_start.date()} → {train_end.date()}")
        logger.info(f"Purge: {train_end.date()} → {test_start.date()}")
        logger.info(f"Test:  {test_start.date()} → {test_end.date()}")
        
        # Create SAC configuration with date-filtered training
        config = self.create_sac_config(fold, train_start, train_end)
        
        # Build algorithm
        algo = config.build()
        
        # Training loop
        total_timesteps = self.config.warmup_timesteps + self.config.finetune_timesteps
        
        logger.info(f"\nTraining for {total_timesteps} timesteps...")
        logger.info(f"Phase 1: 0-{self.config.warmup_timesteps} (Actor frozen)")
        logger.info(f"Phase 2: {self.config.warmup_timesteps}+ (Actor unfrozen)")
        
        # Train using algo.train()
        results = []
        for i in range(total_timesteps // 1000):  # Train in iterations
            result = algo.train()
            results.append(result)
            
            timesteps = result.get("timesteps_total", 0)
            phase = result.get("phase", "unknown")
            
            if i % 10 == 0:  # Log every 10 iterations
                logger.info(f"Iteration {i}: {timesteps} timesteps, Phase: {phase}")
            
            if timesteps >= total_timesteps:
                break
        
        # Get final training metrics
        final_result = results[-1] if results else {}
        
        # ====================================================================
        # EXHAUSTIVE OUT-OF-SAMPLE EVALUATION
        # ====================================================================
        # Create SEPARATE evaluation environment with test date constraints
        # to prevent any training data leakage into evaluation.
        
        logger.info(f"\n{'='*70}")
        logger.info(f"EXHAUSTIVE TEST SET EVALUATION - Fold {fold}")
        logger.info(f"{'='*70}")
        
        # Extract trained policy
        policy = algo.get_policy()
        
        # CRITICAL: Create separate evaluation environment with test date constraints
        # This ensures ZERO leakage from training data
        test_start_str = test_start.strftime('%Y-%m-%d')
        test_end_str = test_end.strftime('%Y-%m-%d')
        
        logger.info(f"Test window: {test_start.date()} to {test_end.date()}")
        
        # Create isolated eval environment with mode="eval"
        eval_env_config = {
            "initial_capital": 100000.0,
            "date_range": (test_start_str, test_end_str),  # STRICT test bounds
            "seed": fold * 1000 + 500,  # Different seed from training
            "mode": "eval",  # CRITICAL: Explicit eval mode
        }
        
        # Import here to avoid circular dependency
        from src.rl.env import ParabolicReversalEnv
        
        eval_env = ParabolicReversalEnv(config=eval_env_config)
        logger.info(f"[EVAL] Created isolated evaluation environment")
        
        # Get test setups for sequential evaluation
        # POLICY: Parquet data takes priority over CSV when (symbol, date) collides.
        # We concatenate parquet first, then CSV, and keep the first occurrence.
        test_setups = (
            eval_env.data_provider.parquet_setups + 
            eval_env.data_provider.csv_setups
        )
        
        # DEDUPLICATE by (symbol, date) to prevent double-counting.
        # "First occurrence wins" - parquet preferred over CSV due to richer 
        # metadata (full OHLCV bars vs. summary statistics).
        seen = set()
        unique_test_setups = []
        for setup in test_setups:
            key = (setup['symbol'], setup['date'])
            if key not in seen:
                seen.add(key)
                unique_test_setups.append(setup)
        test_setups = unique_test_setups
        
        # Sort for deterministic evaluation
        test_setups = sorted(test_setups, key=lambda x: (x['date'], x['symbol']))
        
        logger.info(f"Total test episodes available: {len(test_setups)}")
        
        if len(test_setups) == 0:
            logger.error("No test episodes found in window!")
            test_metrics = {
                'episodes_requested': 0,
                'episodes_evaluated': 0,
                'episodes_failed_to_load': 0,
                'total_test_pnl': 0,
                'mean_episode_pnl': None,
                'median_episode_pnl': None,
                'min_episode_pnl': None,
                'max_episode_pnl': None,
                'win_rate': None,
                'winning_episodes': 0,
                'losing_episodes': 0,
                'total_trades': 0,
                'mean_trades_per_episode': None,
                'per_episode_results': []
            }
        else:
            # Run EXHAUSTIVE evaluation over ALL test setups
            episodes_requested = len(test_setups)
            episodes_evaluated = 0
            episodes_failed_to_load = 0
            episode_results = []
            
            for episode_idx, setup in enumerate(test_setups, 1):
                symbol = setup['symbol']
                date_str = setup['date']
                
                # CRITICAL: Reset environment with fixed_setup option
                obs, info = eval_env.reset(options={
                    "fixed_setup": {"symbol": symbol, "date": date_str}
                })
                
                # Check if episode loaded successfully
                loaded_symbol = eval_env.data_provider.current_symbol
                loaded_date = eval_env.data_provider.current_date
                
                if loaded_symbol is None:
                    logger.warning(f"  [{episode_idx}/{len(test_setups)}] FAILED TO LOAD: "
                                  f"{symbol} {date_str} - skipping")
                    episodes_failed_to_load += 1
                    continue
                
                if loaded_symbol != symbol or loaded_date != date_str:
                    logger.warning(f"  [{episode_idx}/{len(test_setups)}] MISMATCH: "
                                  f"requested {symbol}/{date_str}, got {loaded_symbol}/{loaded_date}")
                    episodes_failed_to_load += 1
                    continue
                
                done = False
                truncated = False
                step_count = 0
                max_steps = 500
                
                while not (done or truncated) and step_count < max_steps:
                    obs_dict = obs if isinstance(obs, dict) else {'state': obs}
                    
                    action, _, _ = policy.compute_single_action(
                        obs_dict,
                        explore=False
                    )
                    
                    obs, reward, done, truncated, info = eval_env.step(action)
                    step_count += 1
                
                episodes_evaluated += 1
                episode_pnl = eval_env.episode_pnl
                episode_trades = eval_env.episode_trades
                
                episode_results.append({
                    'symbol': symbol,
                    'date': date_str,
                    'pnl': episode_pnl,
                    'trades': episode_trades,
                    'steps': step_count
                })
                
                if episode_idx % 10 == 0 or episode_idx <= 5:
                    logger.info(f"  [{episode_idx}/{len(test_setups)}] {symbol} {date_str} | "
                               f"PnL: ${episode_pnl:,.2f} | Trades: {episode_trades}")
            
            # Compute honest metrics
            pnls = [e['pnl'] for e in episode_results]
            winning = sum(1 for p in pnls if p > 0)
            
            test_metrics = {
                'episodes_requested': episodes_requested,
                'episodes_evaluated': episodes_evaluated,
                'episodes_failed_to_load': episodes_failed_to_load,
                'total_test_pnl': sum(pnls) if pnls else 0,
                'mean_episode_pnl': float(np.mean(pnls)) if pnls else None,
                'median_episode_pnl': float(np.median(pnls)) if pnls else None,
                'min_episode_pnl': min(pnls) if pnls else None,
                'max_episode_pnl': max(pnls) if pnls else None,
                'win_rate': winning / episodes_evaluated if episodes_evaluated > 0 else None,
                'winning_episodes': winning if episodes_evaluated > 0 else 0,
                'losing_episodes': episodes_evaluated - winning if episodes_evaluated > 0 else 0,
                'total_trades': sum(e['trades'] for e in episode_results),
                'mean_trades_per_episode': float(np.mean([e['trades'] for e in episode_results])) if episode_results else None,
                'per_episode_results': episode_results
            }
            
            if episodes_evaluated > 0:
                logger.info(f"\nTest Summary: {episodes_evaluated}/{episodes_requested} episodes | "
                           f"PnL: ${test_metrics['total_test_pnl']:,.2f} | "
                           f"Win Rate: {winning}/{episodes_evaluated} | "
                           f"Trades: {test_metrics['total_trades']}")
            else:
                logger.error(f"\nZERO episodes evaluated! "
                            f"({episodes_failed_to_load}/{episodes_requested} failed to load)")
        
        # Save checkpoint
        checkpoint_dir = Path(self.config.output_dir) / f"fold_{fold}_checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        algo.save_checkpoint(str(checkpoint_dir))
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
        
        # Compile results
        result_data = {
            'fold': fold,
            'train_start': train_start.isoformat(),
            'train_end': train_end.isoformat(),
            'test_start': test_start.isoformat(),
            'test_end': test_end.isoformat(),
            'timesteps_total': final_result.get('timesteps_total', 0),
            'phase_at_end': final_result.get('phase', 'unknown'),
            'train_timesteps_target': self.config.warmup_timesteps + self.config.finetune_timesteps,
            'train_seed': fold * 1000,
            'eval_seed': fold * 1000 + 500,
            'evaluation_methodology': 'exhaustive_deterministic',
            'test_metrics': test_metrics
        }
        
        # Cleanup
        algo.stop()
        
        return result_data
    
    def _extract_metrics(self, eval_results: Dict) -> Dict:
        """Extract relevant metrics from evaluation results."""
        if 'evaluation' in eval_results:
            return eval_results['evaluation']
        return eval_results
    
    def run(self):
        """Run complete Walk-Forward Optimization."""
        
        # Setup WFO splits
        splitter = WalkForwardSplitter(
            start_date=datetime(2020, 7, 27),
            end_date=datetime(2024, 12, 30),
            train_years=self.config.train_years,
            test_months=self.config.test_months,
            purge_days=self.config.purge_days,
            step_months=self.config.step_months
        )
        
        logger.info(f"\n{'='*70}")
        logger.info(f"WALK-FORWARD OPTIMIZATION WITH RAY RLLIB")
        logger.info(f"{'='*70}")
        logger.info(f"Total folds: {len(splitter)}")
        logger.info(f"Train window: {self.config.train_years} years")
        logger.info(f"Test window: {self.config.test_months} months")
        logger.info(f"Purge period: {self.config.purge_days} days")
        logger.info(f"Phase 1 (Actor frozen): {self.config.warmup_timesteps} timesteps")
        logger.info(f"Phase 2 (Actor unfrozen): {self.config.finetune_timesteps} timesteps")
        logger.info(f"{'='*70}\n")
        
        # Run each fold
        all_results = []
        for split in splitter:
            result = self.train_fold(
                fold=split['fold'],
                train_start=split['train_start'],
                train_end=split['train_end'],
                test_start=split['test_start'],
                test_end=split['test_end']
            )
            all_results.append(result)
        
        # Aggregate results
        logger.info(f"\n{'='*70}")
        logger.info(f"WFO COMPLETE - AGGREGATED RESULTS")
        logger.info(f"{'='*70}")
        
        valid_folds = [r for r in all_results 
                      if r['test_metrics']['episodes_evaluated'] > 0]
        
        if len(valid_folds) == 0:
            logger.error("NO VALID FOLDS - cannot compute aggregates")
            aggregate_metrics = {
                'error': 'No folds with valid evaluations',
                'total_folds': len(all_results),
                'valid_folds': 0
            }
        else:
            fold_totals = [r['test_metrics']['total_test_pnl'] for r in valid_folds]
            
            aggregate_metrics = {
                'total_folds': len(all_results),
                'valid_folds': len(valid_folds),
                'mean_of_fold_totals': float(np.mean(fold_totals)),
                'total_pnl_across_all_folds': sum(fold_totals),
                'total_episodes_evaluated': sum(r['test_metrics']['episodes_evaluated'] for r in valid_folds),
            }
            
            logger.info(f"Mean of Fold Totals: ${aggregate_metrics['mean_of_fold_totals']:,.2f}")
            logger.info(f"Total PnL: ${aggregate_metrics['total_pnl_across_all_folds']:,.2f}")
        
        # Save results
        results_path = Path(self.config.output_dir) / "wfo_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'run_config': {
                    'train_years': self.config.train_years,
                    'test_months': self.config.test_months,
                    'purge_days': self.config.purge_days,
                    'warmup_timesteps': self.config.warmup_timesteps,
                    'finetune_timesteps': self.config.finetune_timesteps,
                },
                'per_fold_results': all_results,
                'aggregate': aggregate_metrics
            }, f, indent=2, default=str)
        
        logger.info(f"\nResults saved to {results_path}")
        
        # Shutdown Ray
        ray.shutdown()
        
        return all_results


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for WFO training with Ray RLlib."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Walk-Forward Optimization with Ray RLlib')
    parser.add_argument('--bc-checkpoint', type=str,
                        default='models/behavioral_cloning/bc_actor_rllib.pt')
    parser.add_argument('--warmup-steps', type=int, default=20000)
    parser.add_argument('--finetune-steps', type=int, default=50000)
    parser.add_argument('--train-years', type=int, default=2)
    parser.add_argument('--test-months', type=int, default=6)
    parser.add_argument('--purge-days', type=int, default=10)
    parser.add_argument('--output-dir', type=str, default='models/wfo')
    
    args = parser.parse_args()
    
    config = WFOConfig(
        bc_checkpoint=args.bc_checkpoint,
        warmup_timesteps=args.warmup_steps,
        finetune_timesteps=args.finetune_steps,
        train_years=args.train_years,
        test_months=args.test_months,
        purge_days=args.purge_days,
        output_dir=args.output_dir
    )
    
    trainer = WalkForwardRLlibTrainer(config)
    results = trainer.run()
    
    return results


if __name__ == "__main__":
    main()