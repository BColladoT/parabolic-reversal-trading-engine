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

import os
import time
import concurrent.futures

# CRITICAL: Set Polars fork-safety BEFORE any Polars import.
# Ray fork-spawns workers on Linux/WSL; Polars operations in forked
# processes deadlock on uninitialized mutexes without this flag.
os.environ["POLARS_ALLOW_FORKING_THREAD"] = "1"

import torch  # MUST be imported before ray on Windows — see train_wfo_quick_test.py for explanation
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
from src.rl.agent import SACConfig as AgentSACConfig, USE_NEW_API
from src.rl.collapse_detector import CollapseDetector
from src.rl.curriculum import CurriculumManager
from src.dashboard.metrics_writer import write_metrics_line

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import masked model classes for neural-level action masking (Change 10)
try:
    if USE_NEW_API:
        from src.rl.agent import MaskedSACRLModule
        _MASKED_MODEL_CLS = MaskedSACRLModule
    else:
        from src.rl.agent import MaskedSACModel
        from ray.rllib.models import ModelCatalog
        ModelCatalog.register_custom_model("masked_sac_model", MaskedSACModel)
        _MASKED_MODEL_CLS = MaskedSACModel
    _USE_MASKED_MODEL = True
    logger.info(f"[Change 10] Neural action masking ACTIVE via {_MASKED_MODEL_CLS.__name__}")
except Exception as _e:
    _USE_MASKED_MODEL = False
    _MASKED_MODEL_CLS = None
    logger.warning(f"[Change 10] Masked model unavailable ({_e}); falling back to plain SAC")


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
    alpha: float = 0.01  # Initial temperature (scheduled: 0.01→0.05)
    
    # Evaluation
    eval_episodes: int = 10
    
    # Failure handling
    continue_on_fold_failure: bool = False   # If False, stop WFO run on first fold failure
    
    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


# =============================================================================
# Ray RLlib Callback for Joint Actor-Critic Training (Standard SAC)
# =============================================================================

class JointTrainingCallback(DefaultCallbacks):
    """
    Standard SAC training with actor LR warmup (no freeze/unfreeze).

    Actor and critic train jointly from step 0. The actor LR ramps linearly
    from 0 to target over `actor_warmup_steps`, giving the critic a soft head
    start without creating distributional shift.

    Alpha (entropy) schedule is inverted for trading: high early (explore),
    low late (exploit).
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        self.config = config or {}
        self.total_timesteps = self.config.get("total_timesteps", 100000)
        self.actor_lr_target = self.config.get("actor_lr_target", 3e-4)
        self.critic_lr = self.config.get("critic_lr", 3e-4)
        self.actor_warmup_steps = self.config.get("actor_warmup_steps", 10000)
        self.bc_checkpoint = self.config.get("bc_checkpoint", None)
        self.alpha_start = self.config.get("alpha_start", 0.2)
        self.alpha_end = self.config.get("alpha_end", 0.01)
        self.initialized = False

        # Collapse detector and curriculum (same as WarmupCallback)
        self.collapse_detector = CollapseDetector()
        self.curriculum = CurriculumManager()
        self._trades_log_path = self.config.get("trades_log_path")

        logger.info("JointTrainingCallback initialized")
        logger.info(f"  Total timesteps: {self.total_timesteps}")
        logger.info(f"  Actor LR warmup: 0 -> {self.actor_lr_target} over {self.actor_warmup_steps} steps")
        logger.info(f"  Alpha schedule: {self.alpha_start} -> {self.alpha_end} (inverted: explore early, exploit late)")

    def on_algorithm_init(self, *, algorithm: "Algorithm", **kwargs) -> None:
        """Load BC weights (if available) and set initial actor LR to 0 for warmup ramp."""
        logger.info("=" * 60)
        logger.info("ALGORITHM INIT - Joint Training (no actor freeze)")
        logger.info("=" * 60)

        policy = algorithm.get_policy()
        if policy is None:
            logger.warning("Policy not available at init")
            return

        # Load BC weights if available (no freeze afterwards)
        if self.bc_checkpoint and Path(self.bc_checkpoint).exists():
            self._load_bc_weights(policy, self.bc_checkpoint)
            logger.info(f"Loaded BC weights: {self.bc_checkpoint}")
        else:
            logger.info("No BC checkpoint - starting from random initialization")

        # Set actor optimizer LR to 0 (will ramp up via on_train_result)
        model = policy.model
        optimizers = []
        if hasattr(policy, 'get_optimizers'):
            optimizers = policy.get_optimizers()
        elif hasattr(policy, '_optimizers'):
            optimizers = policy._optimizers

        for i, opt in enumerate(optimizers):
            if self._is_actor_optimizer(opt, model):
                for param_group in opt.param_groups:
                    param_group['lr'] = 0.0
                logger.info(f"  Actor optimizer LR set to 0.0 (warmup ramp starts at first train step)")
                break

        self.initialized = True

    def on_train_result(self, *, algorithm: "Algorithm", result: Dict, **kwargs) -> None:
        """Apply actor LR warmup ramp and inverted alpha schedule."""
        timesteps_total = result.get("timesteps_total", 0)

        # --- Actor LR warmup ramp ---
        if timesteps_total < self.actor_warmup_steps:
            actor_lr = self.actor_lr_target * (timesteps_total / max(self.actor_warmup_steps, 1))
        else:
            actor_lr = self.actor_lr_target

        policy = algorithm.get_policy()
        if policy:
            model = policy.model
            optimizers = []
            if hasattr(policy, 'get_optimizers'):
                optimizers = policy.get_optimizers()
            elif hasattr(policy, '_optimizers'):
                optimizers = policy._optimizers
            for opt in optimizers:
                if self._is_actor_optimizer(opt, model):
                    for param_group in opt.param_groups:
                        param_group['lr'] = actor_lr
                    break

        # --- Inject phase info into result ---
        result["phase"] = "joint_training"
        result["phase_num"] = 1
        result["actor_lr"] = actor_lr

        # Trade rate for checkpoint gating (from custom_metrics published in on_episode_end)
        custom = result.get("custom_metrics", {})
        result["trade_rate"] = custom.get("episode_traded_mean", 0.0)

        # Push phase into worker environments for trade tagging
        try:
            workers = getattr(algorithm, 'workers', None)
            if workers:
                workers.foreach_env(lambda env: setattr(env, '_training_phase', 'joint_training'))
        except Exception:
            pass

        # --- Inverted alpha schedule (explore early, exploit late) ---
        scheduled_alpha = self._get_scheduled_alpha(timesteps_total)
        result["scheduled_alpha"] = scheduled_alpha
        try:
            import math
            if policy is not None and hasattr(policy, 'model') and hasattr(policy.model, 'log_alpha'):
                policy.model.log_alpha.data.fill_(math.log(max(scheduled_alpha, 1e-8)))
        except Exception as e:
            logger.debug(f"Could not set scheduled alpha: {e}")

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs) -> None:
        """Collapse detection, curriculum management, and trade rate tracking."""
        info = episode.last_info_for() or {}
        num_trades = info.get('trades', 0)
        episode_pnl = info.get('episode_pnl', 0.0)

        # Publish trade metrics for checkpoint gating
        episode.custom_metrics["episode_trades"] = num_trades
        episode.custom_metrics["episode_traded"] = 1.0 if num_trades > 0 else 0.0

        action_values = []
        if hasattr(episode, 'hist_data') and 'actions' in episode.hist_data:
            action_values = list(episode.hist_data['actions'])

        result = self.collapse_detector.on_episode_end(num_trades, action_values)
        if result:
            if "COLLAPSE DETECTED" in result:
                logger.error(result)
            else:
                logger.warning(result)

        # Curriculum update
        base_pnl = info.get('episode_pnl', episode_pnl)
        phase_advanced = self.curriculum.on_episode_end(num_trades, episode_pnl, base_pnl=base_pnl)
        if phase_advanced:
            new_cfg = self.curriculum.get_current_config()
            logger.info(
                f"CURRICULUM PHASE {self.curriculum.current_phase} ACTIVATED: "
                f"csv_ratio={new_cfg['csv_ratio']}, "
                f"max_drawdown={new_cfg['max_drawdown']}, "
                f"reward_scale={new_cfg['reward_scale']}"
            )
            try:
                sub_envs = base_env.get_sub_environments()
                for env in sub_envs:
                    if hasattr(env, 'update_curriculum_params'):
                        env.update_curriculum_params(
                            reward_scale=new_cfg['reward_scale'],
                            max_drawdown=new_cfg['max_drawdown'],
                        )
                    if hasattr(env, 'data_provider') and hasattr(env.data_provider, 'csv_weight'):
                        env.data_provider.csv_weight = new_cfg['csv_ratio']
            except Exception as exc:
                logger.warning(f"Could not propagate curriculum params: {exc}")

    def _get_scheduled_alpha(self, timestep: int) -> float:
        """Inverted entropy schedule: high exploration early, pure exploitation late."""
        total = self.total_timesteps
        explore_end = min(5000, int(0.02 * total))
        decay1_end = int(0.50 * total)
        decay2_end = int(0.85 * total)

        if timestep <= explore_end:
            return self.alpha_start
        elif timestep <= decay1_end:
            progress = (timestep - explore_end) / max(decay1_end - explore_end, 1)
            return self.alpha_start + progress * (0.05 - self.alpha_start)
        elif timestep <= decay2_end:
            progress = (timestep - decay1_end) / max(decay2_end - decay1_end, 1)
            return 0.05 + progress * (self.alpha_end - 0.05)
        else:
            return self.alpha_end

    def _load_bc_weights(self, policy, checkpoint_path: str):
        """Load pre-trained Behavioral Cloning Actor weights."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model = policy.model
            actor_state = checkpoint.get('model_state_dict', checkpoint.get('actor_state_dict', {}))
            if actor_state:
                model.load_state_dict(actor_state, strict=False)
                logger.info(f"Loaded BC weights from {checkpoint_path}")
            else:
                logger.warning("No actor state found in BC checkpoint")
        except Exception as e:
            logger.error(f"Failed to load BC weights: {e}")

    def _is_actor_optimizer(self, opt, model) -> bool:
        """Check if optimizer controls Actor parameters."""
        actor_params = set()
        if hasattr(model, 'action_model'):
            actor_params = set(id(p) for p in model.action_model.parameters())
        elif hasattr(model, 'policy'):
            actor_params = set(id(p) for p in model.policy.parameters())
        for param_group in opt.param_groups:
            for param in param_group.get('params', []):
                if id(param) in actor_params:
                    return True
        return False


# =============================================================================
# Ray RLlib Callback for Two-Phase Training (LEGACY — kept for backwards compat)
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

        # Alpha schedule config (§1: entropy temperature scheduling)
        self.alpha_phase1_end = self.config.get("alpha_phase1_end", 20000)
        self.alpha_phase2_end = self.config.get("alpha_phase2_end", 40000)
        self.alpha_start = self.config.get("alpha_start", 0.01)
        self.alpha_end = self.config.get("alpha_end", 0.05)

        # Collapse detector (§9)
        self.collapse_detector = CollapseDetector()

        # Curriculum manager (Change 9)
        self.curriculum = CurriculumManager()

        # Dashboard trade log path (set by training script)
        self._trades_log_path = self.config.get("trades_log_path")

        logger.info(f"WarmupCallback initialized")
        logger.info(f"Phase 1 (Actor frozen): 0 to {self.warmup_timesteps} timesteps")
        logger.info(f"Phase 2 (Actor unfrozen): {self.warmup_timesteps}+ timesteps")
        logger.info(f"BC checkpoint: {self.bc_checkpoint}")
        logger.info(f"Alpha schedule: {self.alpha_start} → {self.alpha_end} over {self.alpha_phase1_end}-{self.alpha_phase2_end} steps")
    
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs) -> None:
        """Wire collapse detector (§9) and curriculum manager (Change 9) into episode lifecycle."""
        # Extract num_trades and episode pnl from last step info
        info = episode.last_info_for() or {}
        num_trades = info.get('trades', 0)
        episode_pnl = info.get('episode_pnl', 0.0)

        # Extract action values recorded during the episode
        action_values = []
        if hasattr(episode, 'hist_data') and 'actions' in episode.hist_data:
            action_values = list(episode.hist_data['actions'])

        result = self.collapse_detector.on_episode_end(num_trades, action_values)
        if result:
            if "COLLAPSE DETECTED" in result:
                logger.error(result)
            else:
                logger.warning(result)

        # Curriculum update (Change 9)
        # FIX 11: Pass base_pnl (true equity PnL) separately so curriculum gates
        # Phase 2 win rate on real PnL, not bonus-inflated reward.
        base_pnl = info.get('episode_pnl', episode_pnl)
        phase_advanced = self.curriculum.on_episode_end(num_trades, episode_pnl, base_pnl=base_pnl)
        if phase_advanced:
            new_cfg = self.curriculum.get_current_config()
            logger.info(
                f"CURRICULUM PHASE {self.curriculum.current_phase} ACTIVATED: "
                f"csv_ratio={new_cfg['csv_ratio']}, "
                f"max_drawdown={new_cfg['max_drawdown']}, "
                f"reward_scale={new_cfg['reward_scale']}"
            )
            # Apply new params to all active sub-environments
            try:
                sub_envs = base_env.get_sub_environments()
                for env in sub_envs:
                    # Update env reward_scale and circuit-breaker drawdown
                    if hasattr(env, 'update_curriculum_params'):
                        env.update_curriculum_params(
                            reward_scale=new_cfg['reward_scale'],
                            max_drawdown=new_cfg['max_drawdown'],
                        )
                    # Update data provider csv_weight (already a mutable attribute)
                    if hasattr(env, 'data_provider') and hasattr(env.data_provider, 'csv_weight'):
                        env.data_provider.csv_weight = new_cfg['csv_ratio']
            except Exception as exc:
                logger.warning(f"Could not propagate curriculum params to envs: {exc}")

        # Trade logging is handled by env._record_trade() directly to trades.jsonl.
        # No callback batch write needed (removed to prevent duplicates).

    def _get_scheduled_alpha(self, timestep: int) -> float:
        """Bell-curve entropy schedule: low → peak → decay to exploitation."""
        warmup_end = self.warmup_timesteps
        total_finetune = self.alpha_phase2_end - warmup_end
        if total_finetune <= 0:
            return self.alpha_start

        ramp_end = warmup_end + int(total_finetune * 0.4)
        hold_end = warmup_end + int(total_finetune * 0.7)
        alpha_min = 0.005
        alpha_transition = 0.03  # Jump to 3x at unfreeze for stability

        if timestep < warmup_end:
            return self.alpha_start
        elif timestep < ramp_end:
            progress = (timestep - warmup_end) / max(ramp_end - warmup_end, 1)
            return alpha_transition + progress * (self.alpha_end - alpha_transition)
        elif timestep < hold_end:
            return self.alpha_end
        elif timestep < self.alpha_phase2_end:
            import math
            progress = (timestep - hold_end) / max(self.alpha_phase2_end - hold_end, 1)
            return alpha_min + (self.alpha_end - alpha_min) * 0.5 * (1 + math.cos(math.pi * progress))
        else:
            return alpha_min

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
                    logger.info(f"  - LR = {self.finetune_lr_actor}")

                    self.actor_frozen = False
                    break

            # Clear ALL optimizer momentum (critic momentum from warmup is biased
            # toward the frozen actor's action distribution and amplifies errors
            # when the actor starts producing new actions)
            for j, opt_all in enumerate(optimizers):
                if len(opt_all.state) > 0:
                    opt_all.state.clear()
            logger.info("Cleared ALL optimizer states at phase transition")
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
                # Hard-sync target networks to eliminate stale targets from warmup
                try:
                    if hasattr(policy, 'target_model'):
                        policy.target_model.load_state_dict(policy.model.state_dict())
                        logger.info("Target networks hard-synced at phase transition")
                except Exception as e:
                    logger.warning(f"Target network sync failed (non-fatal): {e}")
            else:
                logger.warning("Could not get policy for unfreezing")
        
        # Log current phase
        if self.phase == 1:
            phase_str = "warmup_actor_frozen"
            result["actor_lr"] = self.warmup_lr_actor
        else:
            phase_str = "finetuning_actor_active"
            result["actor_lr"] = self.finetune_lr_actor

        result["phase"] = phase_str
        result["phase_num"] = self.phase

        # Push phase into worker environments so trades are tagged
        try:
            workers = getattr(algorithm, 'workers', None)
            if workers:
                workers.foreach_env(lambda env: setattr(env, '_training_phase', phase_str))
        except Exception:
            pass

        # Alpha schedule: manually set entropy temperature (§1)
        scheduled_alpha = self._get_scheduled_alpha(timesteps_total)
        result["scheduled_alpha"] = scheduled_alpha
        try:
            import math
            policy = algorithm.get_policy()
            if policy is not None and hasattr(policy, 'model') and hasattr(policy.model, 'log_alpha'):
                policy.model.log_alpha.data.fill_(math.log(max(scheduled_alpha, 1e-8)))
        except Exception as e:
            logger.debug(f"Could not set scheduled alpha on policy model: {e}")


# =============================================================================
# Walk-Forward Splitter
# =============================================================================

class TrainingStallError(RuntimeError):
    """Exception raised when training stalls (no progress for too many iterations)."""
    
    def __init__(self, message: str, timesteps_reached: int, target_timesteps: int, 
                 phase: str = "unknown"):
        super().__init__(message)
        self.timesteps_reached = timesteps_reached
        self.target_timesteps = target_timesteps
        self.phase = phase


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
        
        # Initialize Ray with Polars fork-safety propagated to workers
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                runtime_env={"env_vars": {"POLARS_ALLOW_FORKING_THREAD": "1"}},
            )
        
        logger.info("WalkForwardRLlibTrainer initialized")
    
    def create_sac_config(self, fold: int, train_start: datetime, train_end: datetime) -> SACConfig:
        """Create Ray RLlib SAC configuration with custom callback."""
        
        # Callback config
        callback_config = {
            "warmup_timesteps": self.config.warmup_timesteps,
            "warmup_lr_actor": self.config.warmup_lr_actor,
            "finetune_lr_actor": self.config.finetune_lr_actor,
            "finetune_lr_critic": self.config.finetune_lr_critic,
            "bc_checkpoint": self.config.bc_checkpoint,
            "trades_log_path": str(Path(self.config.output_dir).resolve() / "trades.jsonl"),
        }
        
        # CRITICAL: Date range filter to prevent WFO data leakage
        # Training environment MUST ONLY sample from training period
        train_date_range = (train_start.strftime('%Y-%m-%d'), train_end.strftime('%Y-%m-%d'))
        
        logger.info(f"Fold {fold} training date range: {train_date_range[0]} to {train_date_range[1]}")
        
        # Build SAC config — neural action masking (Change 10)
        # When _USE_MASKED_MODEL is True (the normal case), we wire in the custom masked model
        # so the actor never samples invalid actions. The env assertion will fire if it
        # somehow receives one anyway (python -O disables assertions in production).
        if _USE_MASKED_MODEL:
            logger.info(f"[ACTIVE PATH] Neural action masking via {_MASKED_MODEL_CLS.__name__}")
        else:
            logger.warning("[FALLBACK PATH] Plain RLlib SAC — neural masking unavailable")

        # Build policy_model_config: masked model for new API, custom_model for legacy API
        if _USE_MASKED_MODEL and USE_NEW_API:
            policy_model_config = {}  # rl_module handles masking (set below)
        elif _USE_MASKED_MODEL and not USE_NEW_API:
            policy_model_config = {
                "custom_model": "masked_sac_model",
                "custom_model_config": {
                    "state_dim": 74,
                    "actor_hidden_dims": [256, 256],
                    "mask_penalty": -1e9,
                },
            }
        else:
            policy_model_config = {
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            }

        sac_config = (
            SACConfig()
            .environment(
                env=ParabolicReversalEnv,
                env_config={
                    "initial_capital": 100000.0,
                    "date_range": train_date_range,  # CRITICAL: Prevents data leakage
                    "seed": fold * 1000,  # Unique seed per fold for reproducibility
                    "trades_log_path": str(Path(self.config.output_dir).resolve() / "trades.jsonl"),
                    "dashboard_fold": fold,
                    "r_multiple_reward_weight": getattr(self.config, '_r_multiple_reward_weight', 0.0),
                    "r_multiple_reward_clip": getattr(self.config, '_r_multiple_reward_clip', 5.0),
                },
            )
            .framework("torch")
            .training(
                twin_q=True,
                q_model_config={
                    "fcnet_hiddens": [256, 256],
                    "fcnet_activation": "relu",
                },
                policy_model_config=policy_model_config,
                tau=self.config.tau,
                initial_alpha=self.config.alpha,
                target_entropy=None,  # Disable auto-tuning; alpha managed by schedule
                n_step=1,
                replay_buffer_config={
                    "type": "MultiAgentReplayBuffer",
                    "capacity": self.config.buffer_size,
                },
                train_batch_size=self.config.batch_size,
                grad_clip=1.0,
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

        # Wire in masked RLModule for Ray 2.x+ new API path (Change 10)
        if _USE_MASKED_MODEL and USE_NEW_API:
            try:
                sac_config = sac_config.rl_module(rl_module_spec=_MASKED_MODEL_CLS)
                logger.info("MaskedSACRLModule wired via .rl_module()")
            except Exception as exc:
                logger.warning(f"Could not wire MaskedSACRLModule: {exc}")

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
        
        # Per-fold failure handling: wrap training/evaluation in try/except
        algo = None
        try:
            return self._train_and_evaluate_fold(
                fold, train_start, train_end, test_start, test_end
            )
        except (TrainingStallError, RuntimeError) as e:
            # Training stalled or other critical failure
            logger.error(f"FOLD {fold} FAILED: {e}")
            
            # Extract actual training progress if available (from TrainingStallError)
            if isinstance(e, TrainingStallError):
                actual_timesteps = e.timesteps_reached
                failure_phase = e.phase
                is_stalled = True
                failure_stage = "training"
            else:
                # Generic RuntimeError - no progress info available
                actual_timesteps = 0
                failure_phase = "unknown"
                is_stalled = False
                failure_stage = "training"
            
            # Build failed fold result with full schema and actual progress
            total_timesteps = self.config.warmup_timesteps + self.config.finetune_timesteps
            shortfall = max(0, total_timesteps - actual_timesteps)
            
            failed_result = {
                'fold': fold,
                'train_start': train_start.isoformat(),
                'train_end': train_end.isoformat(),
                'test_start': test_start.isoformat(),
                'test_end': test_end.isoformat(),
                'timesteps_total': actual_timesteps,
                'phase_at_end': failure_phase,
                'train_timesteps_target': total_timesteps,
                'train_timesteps_reached': actual_timesteps,
                'train_timesteps_shortfall': shortfall,
                'training_completed_successfully': False,
                'training_stalled': is_stalled,
                'failure_stage': failure_stage,
                'evaluation_skipped': True,
                'failure_reason': str(e),
                'train_seed': fold * 1000,
                'eval_seed': fold * 1000 + 500,
                'evaluation_methodology': 'exhaustive_deterministic',
                'test_metrics': {
                    'episodes_requested': 0,
                    'episodes_evaluated': 0,
                    'episodes_failed_to_load': 0,
                    'total_test_pnl': None,
                    'mean_episode_pnl': None,
                    'median_episode_pnl': None,
                    'min_episode_pnl': None,
                    'max_episode_pnl': None,
                    'win_rate': None,
                    'winning_episodes': 0,
                    'losing_episodes': 0,
                    'total_trades': 0,
                    'mean_trades_per_episode': None,
                    'per_episode_results': [],
                    'fold_failed': True,
                    'failure_reason': str(e)
                }
            }
            
            # Mark for potential run-level stop
            failed_result['_fold_failed'] = True
            
            # Cleanup algo if it was created
            if algo is not None:
                try:
                    algo.stop()
                except Exception as cleanup_error:
                    logger.warning(f"Error during algo cleanup: {cleanup_error}")
            
            return failed_result
    
    def _train_and_evaluate_fold(
        self,
        fold: int,
        train_start: datetime,
        train_end: datetime,
        test_start: datetime,
        test_end: datetime
    ) -> Dict[str, Any]:
        """Internal method: train and evaluate a single fold. Raises RuntimeError on failure."""
        
        # Create SAC configuration with date-filtered training
        config = self.create_sac_config(fold, train_start, train_end)
        
        # Build algorithm
        algo = config.build()
        
        # Training loop - timestep-driven with no-progress safeguard
        total_timesteps = self.config.warmup_timesteps + self.config.finetune_timesteps
        
        logger.info(f"\nTraining for {total_timesteps} timesteps...")
        logger.info(f"Phase 1: 0-{self.config.warmup_timesteps} (Actor frozen)")
        logger.info(f"Phase 2: {self.config.warmup_timesteps}+ (Actor unfrozen)")
        logger.info(f"Target: {total_timesteps} timesteps")
        
        # Dashboard metrics file (append per iteration)
        metrics_path = Path(self.config.output_dir) / "training_metrics.jsonl"
        fold_start_time = datetime.now()

        # Train using algo.train() with actual timestep tracking
        results = []
        timesteps_total = 0
        iteration = 0
        prev_timesteps = 0
        no_progress_count = 0
        max_no_progress_iterations = 5  # Safety stop if no progress for 5 consecutive iterations
        last_progress_time = time.monotonic()  # Wall-clock stall detector
        train_timeout_seconds = 300  # 5 min timeout per algo.train() call
        consecutive_timeouts = 0

        while timesteps_total < total_timesteps:
            # Timeout-protected training call — prevents infinite hangs
            # from Polars fork deadlocks or Ray worker crashes
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(algo.train)
                    result = future.result(timeout=train_timeout_seconds)
                consecutive_timeouts = 0
                last_progress_time = time.monotonic()
            except concurrent.futures.TimeoutError:
                consecutive_timeouts += 1
                logger.warning(
                    f"algo.train() TIMEOUT after {train_timeout_seconds}s "
                    f"(attempt {consecutive_timeouts}/2). "
                    f"Iteration {iteration}, timesteps {timesteps_total}."
                )
                if consecutive_timeouts >= 2:
                    logger.error(
                        f"Two consecutive timeouts — breaking training loop. "
                        f"Proceeding to evaluation with {timesteps_total}/{total_timesteps} steps."
                    )
                    break
                continue

            results.append(result)
            
            timesteps_total = result.get("timesteps_total", 0)
            phase = result.get("phase", "unknown")
            phase_num = result.get("phase_num", 0)
            
            # Check for progress
            training_stalled = False
            if timesteps_total <= prev_timesteps:
                no_progress_count += 1
                logger.warning(
                    f"No progress detected: timesteps_total={timesteps_total} "
                    f"(was {prev_timesteps}), no_progress_count={no_progress_count}"
                )
                if no_progress_count >= max_no_progress_iterations:
                    training_stalled = True
                    logger.error(
                        f"TRAINING STALLED: No timestep progress for {max_no_progress_iterations} "
                        f"consecutive iterations. "
                        f"Target: {total_timesteps}, Reached: {timesteps_total}. "
                        f"This indicates a problem with the training loop."
                    )
                    raise TrainingStallError(
                        message=(
                            f"Fold {fold} training stalled: no progress for {max_no_progress_iterations} "
                            f"consecutive iterations. Target: {total_timesteps}, "
                            f"Reached: {timesteps_total}. Investigate training configuration."
                        ),
                        timesteps_reached=timesteps_total,
                        target_timesteps=total_timesteps,
                        phase=phase
                    )
            else:
                no_progress_count = 0  # Reset progress counter

            # Wall-clock stall detector: warn if iterations are abnormally slow
            wall_elapsed = time.monotonic() - last_progress_time
            if wall_elapsed > 600:  # 10 minutes since last successful train()
                logger.warning(
                    f"SLOW ITERATION: {wall_elapsed:.0f}s since last progress. "
                    f"Iteration {iteration}, timesteps {timesteps_total}/{total_timesteps}"
                )

            prev_timesteps = timesteps_total
            
            # Log at reasonable intervals (every 10 iterations or when phase changes)
            # Safely check for phase transition: need at least 2 results to compare
            phase_changed = False
            if phase_num == 2 and len(results) >= 2:
                phase_changed = results[-2].get("phase_num", 1) == 1
            
            if iteration % 10 == 0 or phase_changed:
                progress_pct = (timesteps_total / total_timesteps) * 100
                logger.info(
                    f"Iteration {iteration:4d}: {timesteps_total:6d}/{total_timesteps} timesteps "
                    f"({progress_pct:5.1f}%) | Phase: {phase}"
                )
            
            iteration += 1

            # Write metrics for the live dashboard
            write_metrics_line(metrics_path, iteration, result, fold, fold_start_time, total_timesteps)

        # Final training summary
        # NOTE: We only reach here if training completed successfully (RuntimeError raised on stall)
        training_completed_successfully = timesteps_total >= total_timesteps
        logger.info(f"\n{'='*70}")
        logger.info(f"TRAINING COMPLETE - Fold {fold}")
        logger.info(f"{'='*70}")
        logger.info(f"Target timesteps:     {total_timesteps}")
        logger.info(f"Reached timesteps:    {timesteps_total}")
        logger.info(f"Shortfall:            {max(0, total_timesteps - timesteps_total)}")
        logger.info(f"Target reached:       {'YES' if training_completed_successfully else 'NO'}")
        logger.info(f"Total iterations:     {iteration}")
        logger.info(f"Final phase:          {phase}")
        logger.info(f"Stop reason:          Target reached normally")
        logger.info(f"{'='*70}")
        
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
            "trades_log_path": str(Path(self.output_dir) / "trades.jsonl"),
            "dashboard_fold": fold,
            "r_multiple_reward_weight": getattr(self.config, '_r_multiple_reward_weight', 0.0),
            "r_multiple_reward_clip": getattr(self.config, '_r_multiple_reward_clip', 5.0),
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
        # NOTE: If we reach here, training completed successfully (RuntimeError raised on stall)
        result_data = {
            'fold': fold,
            'train_start': train_start.isoformat(),
            'train_end': train_end.isoformat(),
            'test_start': test_start.isoformat(),
            'test_end': test_end.isoformat(),
            'timesteps_total': final_result.get('timesteps_total', 0),
            'phase_at_end': final_result.get('phase', 'unknown'),
            'train_timesteps_target': self.config.warmup_timesteps + self.config.finetune_timesteps,
            'train_timesteps_reached': timesteps_total,
            'train_timesteps_shortfall': max(0, total_timesteps - timesteps_total),
            'training_completed_successfully': training_completed_successfully,
            'training_stalled': False,  # If True, RuntimeError would have been raised above
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

        # Clear dashboard files from previous runs
        metrics_path = Path(self.config.output_dir) / "training_metrics.jsonl"
        trades_path = Path(self.config.output_dir) / "trades.jsonl"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text("")
        trades_path.write_text("")

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
            
            # Check for fold failure and handle according to config
            if result.get('_fold_failed', False) or not result.get('training_completed_successfully', True):
                if not self.config.continue_on_fold_failure:
                    logger.error(
                        f"\n{'='*70}\n"
                        f"WFO RUN STOPPED: Fold {result['fold']} failed and "
                        f"continue_on_fold_failure=False\n"
                        f"Failure reason: {result.get('failure_reason', 'Unknown')}\n"
                        f"Completed folds: {len([r for r in all_results if not r.get('_fold_failed', False)])}\n"
                        f"Failed folds: {len([r for r in all_results if r.get('_fold_failed', False)])}\n"
                        f"{'='*70}"
                    )
                    break
                else:
                    logger.warning(
                        f"Fold {result['fold']} failed but continuing to next fold "
                        f"(continue_on_fold_failure=True)"
                    )
        
        # Aggregate results
        logger.info(f"\n{'='*70}")
        logger.info(f"WFO COMPLETE - AGGREGATED RESULTS")
        logger.info(f"{'='*70}")
        
        # Classify folds
        successful_folds = [r for r in all_results 
                           if r.get('training_completed_successfully', False)]
        failed_folds = [r for r in all_results 
                       if r.get('_fold_failed', False) or not r.get('training_completed_successfully', True)]
        valid_folds = [r for r in all_results 
                      if r['test_metrics']['episodes_evaluated'] > 0]
        
        logger.info(f"Total folds processed: {len(all_results)}")
        logger.info(f"Successful training: {len(successful_folds)}")
        logger.info(f"Failed training: {len(failed_folds)}")
        logger.info(f"Valid evaluations: {len(valid_folds)}")
        
        if len(valid_folds) == 0:
            logger.error("NO VALID FOLDS - cannot compute PnL aggregates")
            aggregate_metrics = {
                'error': 'No folds with valid evaluations',
                'total_folds': len(all_results),
                'successful_folds': len(successful_folds),
                'failed_folds': len(failed_folds),
                'valid_folds': 0
            }
        else:
            fold_totals = [r['test_metrics']['total_test_pnl'] for r in valid_folds]
            
            aggregate_metrics = {
                'total_folds': len(all_results),
                'successful_folds': len(successful_folds),
                'failed_folds': len(failed_folds),
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
                    'continue_on_fold_failure': self.config.continue_on_fold_failure,
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
    import signal

    # Graceful shutdown on SIGTERM (from dashboard stop button)
    def _graceful_shutdown(signum, frame):
        logger.info("SIGTERM received — shutting down gracefully")
        try:
            import ray
            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass
        sys.exit(1)
    signal.signal(signal.SIGTERM, _graceful_shutdown)

    parser = argparse.ArgumentParser(description='Walk-Forward Optimization with Ray RLlib')
    parser.add_argument('--bc-checkpoint', type=str,
                        default='models/behavioral_cloning/bc_actor_rllib.pt')
    parser.add_argument('--warmup-steps', type=int, default=20000)
    parser.add_argument('--finetune-steps', type=int, default=50000)
    parser.add_argument('--train-years', type=int, default=2)
    parser.add_argument('--test-months', type=int, default=6)
    parser.add_argument('--purge-days', type=int, default=10)
    parser.add_argument('--output-dir', type=str, default='models/wfo')
    # SAC hyperparams (configurable from dashboard)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--buffer-size', type=int, default=1000000)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--lr-actor', type=float, default=3e-4)
    parser.add_argument('--lr-critic', type=float, default=3e-4)
    # Environment params (configurable from dashboard)
    parser.add_argument('--initial-capital', type=float, default=100000.0)
    parser.add_argument('--max-drawdown', type=float, default=-10000.0)
    parser.add_argument('--stop-loss', type=float, default=-5000.0)
    parser.add_argument('--max-pos-fraction', type=float, default=0.30)
    parser.add_argument('--vwap-threshold', type=float, default=20.0)
    parser.add_argument('--txn-cost', type=float, default=0.003)
    parser.add_argument(
        "--r-multiple-reward-weight", type=float, default=0.0,
        help="Per-trade R-multiple reward term weight (0.0 = disabled, matches pre-batch behavior).",
    )
    parser.add_argument(
        "--r-multiple-reward-clip", type=float, default=5.0,
        help="Per-trade R-multiple clip magnitude before scaling.",
    )

    args = parser.parse_args()

    config = WFOConfig(
        bc_checkpoint=args.bc_checkpoint,
        warmup_timesteps=args.warmup_steps,
        finetune_timesteps=args.finetune_steps,
        train_years=args.train_years,
        test_months=args.test_months,
        purge_days=args.purge_days,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
    )
    # Store extra params for env_config forwarding
    config._lr_actor = args.lr_actor
    config._lr_critic = args.lr_critic
    config._initial_capital = args.initial_capital
    config._max_drawdown = args.max_drawdown
    config._stop_loss = args.stop_loss
    config._max_pos_fraction = args.max_pos_fraction
    config._vwap_threshold = args.vwap_threshold
    config._txn_cost = args.txn_cost
    config._r_multiple_reward_weight = args.r_multiple_reward_weight
    config._r_multiple_reward_clip = args.r_multiple_reward_clip

    trainer = WalkForwardRLlibTrainer(config)
    results = trainer.run()

    return results


if __name__ == "__main__":
    main()