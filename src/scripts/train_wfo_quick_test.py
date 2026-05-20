"""
Quick Test Run for WFO Training (1-2 hours)

This script runs a shortened WFO training to verify:
1. Data provider loads trading days correctly
2. Environment runs without errors
3. Agent actually learns (PnL != $0)
4. Checkpoints save properly

Recommended: Run this first before the full training.
"""

import torch  # MUST be imported before ray on Windows — ray's C extensions otherwise poison the DLL search path and torch's c10.dll fails to load
import ray
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from typing import Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import json
import csv
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rl.env import ParabolicReversalEnv
from src.baselines.random_agent import RandomAgent
from src.baselines.naive_short_agent import NaiveShortAgent
from src.baselines.evaluate_baseline import evaluate_baseline_on_fold
from src.utils.statistical_tests import (
    bootstrap_confidence_interval,
    permutation_test,
    format_benchmark_report
)
from src.dashboard.metrics_writer import write_metrics_line
from train_wfo import (
    WarmupCallback, JointTrainingCallback, WalkForwardSplitter, WFOConfig,
    _USE_MASKED_MODEL, _MASKED_MODEL_CLS, USE_NEW_API,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _detect_data_range(csv_path: Path = None) -> tuple:
    """
    Auto-detect available data range from the CSV setups file.

    Returns (min_date: datetime, max_date: datetime, total_setups: int).
    Falls back to a conservative default if CSV is missing.
    """
    if csv_path is None:
        csv_path = Path("reports/all_setups_backtest.csv")

    if not csv_path.exists():
        logger.warning(f"CSV setups file not found at {csv_path}, using fallback dates")
        return datetime(2021, 1, 1), datetime(2024, 12, 30), 0

    dates = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = row.get('Date', '').strip()
            if d:
                dates.append(d)

    if not dates:
        logger.warning("CSV setups file is empty, using fallback dates")
        return datetime(2021, 1, 1), datetime(2024, 12, 30), 0

    min_date = datetime.strptime(min(dates), '%Y-%m-%d')
    max_date = datetime.strptime(max(dates), '%Y-%m-%d')
    logger.info(f"Detected data range: {min_date.date()} → {max_date.date()} ({len(dates)} setups)")
    return min_date, max_date, len(dates)


@dataclass
class QuickTestConfig:
    """Quick test configuration - runs in 1-2 hours."""
    
    output_dir: str = "models/wfo_test"
    
    # Number of walk-forward folds (rolling train/test windows)
    # More folds = better robustness check across market regimes
    n_folds: int = 1

    # Train/test window sizes per fold
    train_months: int = 6
    test_months: int = 2
    purge_days: int = 5
    
    # Joint training (replaces two-phase warmup/finetune)
    total_timesteps: int = 25000
    actor_warmup_steps: int = 2000  # LR ramp duration

    # SAC parameters
    buffer_size: int = 50000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    alpha_start: float = 0.2   # High entropy early (explore)
    alpha_end: float = 0.01    # Low entropy late (exploit)

    eval_episodes: int = 50

    # Backwards compat properties for code that still references these
    @property
    def warmup_timesteps(self):
        return 0

    @property
    def finetune_timesteps(self):
        return self.total_timesteps
    
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
        """Create SAC configuration with joint actor-critic training.

        Only invoked when self.config._algo == 'sac' (the default). The
        JointTrainingCallback wired in below is SAC-specific (it accesses
        the actor optimizer via optimizer_names() to freeze/unfreeze it during
        warmup) and is intentionally not used for PPO — see create_ppo_config.
        """
        algo = getattr(self.config, '_algo', 'sac')
        assert algo == 'sac', f"create_sac_config invoked with algo={algo!r}"

        total_timesteps = self.config.total_timesteps
        lr_actor = getattr(self.config, '_lr_actor', 3e-4)
        lr_critic = getattr(self.config, '_lr_critic', 3e-4)

        callback_config = {
            "total_timesteps": total_timesteps,
            "actor_lr_target": lr_actor,
            "critic_lr": lr_critic,
            "actor_warmup_steps": getattr(self.config, 'actor_warmup_steps', 2000),
            "bc_checkpoint": None,
            "alpha_start": self.config.alpha_start,
            "alpha_end": self.config.alpha_end,
            "trades_log_path": str(Path(self.config.output_dir).resolve() / "trades.jsonl"),
        }
        
        # Use masked model when available (Change 10)
        if _USE_MASKED_MODEL and not USE_NEW_API:
            policy_model_config = {
                "custom_model": "masked_sac_model",
                "custom_model_config": {
                    "state_dim": 74,
                    "actor_hidden_dims": [256, 256],
                    "mask_penalty": -1e9,
                },
            }
        else:
            policy_model_config = {"fcnet_hiddens": [256, 256], "fcnet_activation": "relu"}

        sac_config = (
            SACConfig()
            .environment(
                env=ParabolicReversalEnv,
                env_config={
                    "initial_capital": getattr(self.config, '_initial_capital', 100000.0),
                    "annealer_total_timesteps": total_timesteps,
                    "max_drawdown": getattr(self.config, '_max_drawdown', -10000.0),
                    "circuit_breaker_threshold": getattr(self.config, '_max_drawdown', -10000.0),
                    "intra_step_stop_loss": getattr(self.config, '_stop_loss', -2000.0),
                    "max_position_capital_fraction": getattr(self.config, '_max_pos_fraction', 0.30),
                    "min_vwap_deviation_entry": getattr(self.config, '_vwap_threshold', 20.0),
                    "transaction_cost_per_dollar": getattr(self.config, '_txn_cost', 0.003),
                    "r_multiple_reward_weight": getattr(self.config, '_r_multiple_reward_weight', 0.0),
                    "r_multiple_reward_clip": getattr(self.config, '_r_multiple_reward_clip', 5.0),
                    "mfe_evaporation_penalty_max": getattr(self.config, '_mfe_evap_penalty', 0.0),
                    "hold_band_threshold": getattr(self.config, '_hold_band_threshold', 0.05),
                    "entry_threshold": getattr(self.config, '_entry_threshold', None),
                    "cover_threshold": getattr(self.config, '_cover_threshold', None),
                    "action_space_type": getattr(self.config, '_action_space_type', 'continuous'),
                    "trades_log_path": str(Path(self.config.output_dir).resolve() / "trades.jsonl"),
                    "dashboard_fold": fold,
                },
            )
            .framework("torch")
            .training(
                twin_q=True,
                q_model_config={"fcnet_hiddens": [256, 256], "fcnet_activation": "relu"},
                policy_model_config=policy_model_config,
                tau=self.config.tau,
                initial_alpha=self.config.alpha_start,
                target_entropy=None,  # Disable auto-tuning; alpha managed by schedule
                n_step=3,
                replay_buffer_config={
                    "type": "MultiAgentReplayBuffer",
                    "capacity": self.config.buffer_size,
                },
                train_batch_size=self.config.batch_size,
                grad_clip=1.0,
            )
            # SAC-only: JointTrainingCallback freezes/unfreezes the actor optimizer
            # via algorithm.optimizer_names(). PPO has a single optimizer and uses
            # entropy_coeff for exploration — see create_ppo_config (algo == 'ppo').
            .callbacks(callbacks_class=lambda: JointTrainingCallback(callback_config))
            .rollouts(
                num_rollout_workers=0,
                rollout_fragment_length=200,
            )
            .reporting(
                min_time_s_per_iteration=5,
            )
            .evaluation(
                evaluation_interval=None,  # Disabled: we use manual eval loop for true PnL
            )
        )

        # Wire in masked RLModule for new API path (Change 10)
        if _USE_MASKED_MODEL and USE_NEW_API:
            try:
                sac_config = sac_config.rl_module(rl_module_spec=_MASKED_MODEL_CLS)
            except Exception as exc:
                logger.warning(f"Could not wire MaskedSACRLModule: {exc}")

        return sac_config

    def create_ppo_config(self, fold: int) -> PPOConfig:
        """Create PPO configuration — drop-in alternative to SAC.

        PPO is on-policy with a clipped surrogate objective; its policy update
        is bounded per step and its rollout-time noise comes from the policy's
        own stochastic head rather than a separate SAC exploration term. This
        is the apples-to-apples algorithm swap that holds the action space
        constant (Box(-1, 1)) and lets us compare PPO vs SAC directly.

        Uses RLlib's well-tested PPO defaults (clip_param=0.3, num_sgd_iter=20,
        sgd_minibatch_size=128, entropy_coeff=0.01, lr=3e-4). No BC warm-start
        (MaskedGaussianPolicy actor architecture is not loadable by PPO) and
        no SAC-specific callbacks (PPO uses a single optimizer; SAC's
        JointTrainingCallback freezes/unfreezes the actor optimizer).
        """
        total_timesteps = self.config.total_timesteps

        ppo_config = (
            PPOConfig()
            .environment(
                env=ParabolicReversalEnv,
                env_config={
                    "initial_capital": getattr(self.config, '_initial_capital', 100000.0),
                    "annealer_total_timesteps": total_timesteps,
                    "max_drawdown": getattr(self.config, '_max_drawdown', -10000.0),
                    "circuit_breaker_threshold": getattr(self.config, '_max_drawdown', -10000.0),
                    "intra_step_stop_loss": getattr(self.config, '_stop_loss', -2000.0),
                    "max_position_capital_fraction": getattr(self.config, '_max_pos_fraction', 0.30),
                    "min_vwap_deviation_entry": getattr(self.config, '_vwap_threshold', 20.0),
                    "transaction_cost_per_dollar": getattr(self.config, '_txn_cost', 0.003),
                    "r_multiple_reward_weight": getattr(self.config, '_r_multiple_reward_weight', 0.0),
                    "r_multiple_reward_clip": getattr(self.config, '_r_multiple_reward_clip', 5.0),
                    "mfe_evaporation_penalty_max": getattr(self.config, '_mfe_evap_penalty', 0.0),
                    "hold_band_threshold": getattr(self.config, '_hold_band_threshold', 0.05),
                    "entry_threshold": getattr(self.config, '_entry_threshold', None),
                    "cover_threshold": getattr(self.config, '_cover_threshold', None),
                    "action_space_type": getattr(self.config, '_action_space_type', 'continuous'),
                    "trades_log_path": str(Path(self.config.output_dir).resolve() / "trades.jsonl"),
                    "dashboard_fold": fold,
                },
            )
            .framework("torch")
            .training(
                gamma=self.config.gamma,
                lr=getattr(self.config, '_lr_actor', 3e-4),
                train_batch_size=4000,
                sgd_minibatch_size=128,
                num_sgd_iter=20,
                clip_param=0.3,
                entropy_coeff=0.01,
                vf_clip_param=10.0,
                grad_clip=1.0,
                model={"fcnet_hiddens": [256, 256], "fcnet_activation": "relu"},
            )
            .rollouts(
                num_rollout_workers=0,
                rollout_fragment_length=200,
            )
            .reporting(
                min_time_s_per_iteration=5,
            )
            .evaluation(
                evaluation_interval=None,  # manual eval loop (same path as SAC)
            )
            .resources(num_gpus=1 if torch.cuda.is_available() else 0)
        )

        return ppo_config

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

        algo_choice = getattr(self.config, '_algo', 'sac')
        if algo_choice == 'ppo':
            logger.info("Algorithm: PPO (on-policy, single optimizer, no BC warm-start)")
            config = self.create_ppo_config(fold)
        else:
            logger.info("Algorithm: SAC (off-policy, masked Gaussian, joint training callback)")
            config = self.create_sac_config(fold)
        algo = config.build()
        
        total_timesteps = self.config.total_timesteps

        logger.info(f"\nTraining for {total_timesteps} timesteps (joint actor-critic)...")
        logger.info(f"Actor LR warmup: 0 -> target over first {self.config.actor_warmup_steps} steps")
        logger.info(f"Alpha schedule: {self.config.alpha_start} -> {self.config.alpha_end} (inverted)\n")
        
        # Dashboard metrics file (append per iteration)
        metrics_path = Path(self.config.output_dir) / "training_metrics.jsonl"
        fold_start_time = datetime.now()

        # Best-checkpoint selection (tracks best model during finetuning)
        best_eval_reward = float('-inf')
        best_checkpoint_dir = Path(self.config.output_dir) / f"fold_{fold}_best_checkpoint"
        best_checkpoint_iteration = None
        best_checkpoint_timestep = None

        results = []
        phase_history = []
        alpha_history = []
        last_log_time = datetime.now()
        prev_timesteps = 0
        no_progress_count = 0
        iteration = 0

        # Train until target timesteps reached (adapts to actual step rate)
        while True:
            result = algo.train()
            results.append(result)

            timesteps = result.get("timesteps_total", 0)
            phase = result.get("phase", "unknown")
            phase_history.append(phase)

            alpha_val = result.get("scheduled_alpha", None)
            if alpha_val is not None:
                alpha_history.append((timesteps, alpha_val))

            # Stall detection: abort if no progress for 5 consecutive iterations
            if timesteps <= prev_timesteps:
                no_progress_count += 1
                if no_progress_count >= 5:
                    logger.error(
                        f"TRAINING STALLED: No progress for 5 iterations. "
                        f"Reached {timesteps}/{total_timesteps} steps."
                    )
                    break
            else:
                no_progress_count = 0
            prev_timesteps = timesteps

            # Log every 10 iterations or on phase transition
            if iteration % 10 == 0 or (len(results) >= 2 and results[-2].get("phase") != phase):
                elapsed = (datetime.now() - last_log_time).total_seconds()
                logger.info(
                    f"Iteration {iteration}: {timesteps}/{total_timesteps} steps "
                    f"({timesteps/total_timesteps*100:.0f}%), Phase: {phase}, "
                    f"Time: {elapsed:.1f}s"
                )
                last_log_time = datetime.now()

            iteration += 1

            # Write metrics for the live dashboard
            write_metrics_line(metrics_path, iteration, result, fold, fold_start_time, total_timesteps)

            # Best-checkpoint selection (after warmup, gated by trade rate >10%)
            if timesteps > self.config.actor_warmup_steps:
                eval_reward = result.get("episode_reward_mean", None)
                trade_rate = result.get("trade_rate", 0.0)
                if eval_reward is not None and trade_rate > 0.1 and eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    best_checkpoint_iteration = iteration
                    best_checkpoint_timestep = timesteps
                    try:
                        best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                        algo.save_checkpoint(str(best_checkpoint_dir))
                        logger.info(
                            f"  NEW BEST checkpoint: iter {iteration}, "
                            f"step {timesteps}, reward {eval_reward:.2f}"
                        )
                    except Exception as e:
                        logger.warning(f"  Failed to save best checkpoint: {e}")

            if timesteps >= total_timesteps:
                break

        final_result = results[-1] if results else {}

        # Restore best checkpoint for OOS evaluation
        if best_checkpoint_iteration is not None:
            logger.info(
                f"\nRestoring BEST checkpoint (iter {best_checkpoint_iteration}, "
                f"step {best_checkpoint_timestep}, reward {best_eval_reward:.2f})"
            )
            try:
                algo.restore(str(best_checkpoint_dir))
                logger.info("Best checkpoint restored successfully")
            except Exception as e:
                logger.warning(f"Failed to restore best checkpoint: {e}. Using final model.")
        else:
            logger.warning("No best checkpoint recorded during finetuning. Using final model.")

        # Validate training mechanics
        logger.info(f"\nPipeline Validation:")
        checks = self._validate_training_mechanics(results, phase_history, alpha_history)
        
        # === EVALUATION ON HELD-OUT TEST DATA ===
        # Uses manual episode loop (NOT algo.evaluate()) to get TRUE dollar PnL.
        # algo.evaluate() returns shaped/scaled reward, which is NOT PnL.
        logger.info(f"\nEvaluating on test window (manual loop, true PnL)...")

        test_start_str = test_start.strftime('%Y-%m-%d')
        test_end_str = test_end.strftime('%Y-%m-%d')
        test_setups = self._get_test_setups(
            test_start_str, test_end_str,
            max_episodes=self.config.eval_episodes,
        )

        policy = algo.get_policy()
        eval_env_config = {
            "initial_capital": 100000.0,
            "date_range": (test_start_str, test_end_str),
            "mode": "eval",
            "trades_log_path": str(Path(self.config.output_dir).resolve() / "trades.jsonl"),
            "dashboard_fold": fold,
            # Must match training-time action space so the policy's actions are
            # interpreted correctly. Without this, a Discrete-trained policy
            # would emit ints that the default continuous env clips to +1.0
            # and routes through _discretize_action — silently wrong eval.
            "action_space_type": getattr(self.config, '_action_space_type', 'continuous'),
        }

        # PPO uses RLlib's default FullyConnectedNetwork, which expects a flat
        # obs tensor (`input_dict["obs_flat"]`). For Dict observation spaces the
        # rollout pipeline auto-flattens via a preprocessor — but the manual
        # compute_single_action path bypasses that. Apply the preprocessor here
        # so PPO eval works identically to its training-time obs path. SAC's
        # masked model accepts Dict obs natively, so this is a no-op for SAC.
        algo_choice = getattr(self.config, '_algo', 'sac')
        preprocessor = None
        if algo_choice == 'ppo':
            try:
                from ray.rllib.models.preprocessors import get_preprocessor
                # Use a dummy env's observation space (matches what was passed
                # to PPOConfig.environment(...) at construction time).
                _tmp_env = ParabolicReversalEnv(config=eval_env_config)
                preprocessor_cls = get_preprocessor(_tmp_env.observation_space)
                preprocessor = preprocessor_cls(_tmp_env.observation_space)
                _tmp_env.close() if hasattr(_tmp_env, 'close') else None
            except Exception as exc:
                logger.warning(f"Could not build PPO obs preprocessor: {exc}")
                preprocessor = None

        # Phase 1.2: collect OOS action distribution + time-in-position
        # diagnostics from get_episode_diagnostics() (added in Phase 1.1,
        # env.py; renamed from get_episode_info in Phase 1.1 fix I1).
        # The histogram is only populated for discrete action spaces; for
        # continuous it's all zeros (the int-cast of a Box(-1,1) sample isn't
        # a meaningful bin index). We still collect it uniformly here and
        # gate the action_distribution computation on action_space_type
        # below to avoid emitting a meaningless distribution for continuous.
        action_space_type = getattr(self.config, '_action_space_type', 'continuous')

        episode_results = []
        for ep_idx, setup in enumerate(test_setups):
            try:
                eval_env = ParabolicReversalEnv(config=eval_env_config)
                obs, info = eval_env.reset(options={
                    "fixed_setup": {"symbol": setup['symbol'], "date": setup['date']}
                })
                if eval_env.data_provider.current_symbol is None:
                    continue

                done, truncated, step_count = False, False, 0
                while not (done or truncated) and step_count < 500:
                    if preprocessor is not None:
                        # PPO path: flatten Dict obs to a single tensor matching
                        # the network's expected input layout.
                        policy_input = preprocessor.transform(obs)
                    else:
                        # SAC path: pass Dict obs directly (custom masked model).
                        policy_input = obs if isinstance(obs, dict) else {'state': obs}
                    action, _, _ = policy.compute_single_action(policy_input, explore=False)
                    obs, reward, done, truncated, info = eval_env.step(action)
                    step_count += 1

                ep_info = eval_env.get_episode_diagnostics()
                episode_results.append({
                    'symbol': setup['symbol'],
                    'date': setup['date'],
                    'pnl': eval_env.episode_pnl,
                    'trades': ep_info['n_trades'],
                    'action_histogram': ep_info['action_histogram'],
                    'mean_bars_in_position': ep_info['mean_bars_in_position'],
                    'median_bars_in_position': ep_info['median_bars_in_position'],
                })
            except Exception as e:
                logger.warning(f"  Eval episode {setup['symbol']} {setup['date']} failed: {e}")

        # Compute true test metrics
        pnls = [ep['pnl'] for ep in episode_results]
        total_test_pnl = sum(pnls) if pnls else 0.0
        mean_episode_pnl = float(np.mean(pnls)) if pnls else 0.0
        winning = sum(1 for p in pnls if p > 0)
        episodes_evaluated = len(episode_results)

        # Phase 1.2: aggregate OOS action distribution across all eval
        # episodes in this fold. For continuous action spaces the histograms
        # are all zeros (see env.get_episode_diagnostics docstring); we set
        # action_distribution to None and document via a top-level note so
        # downstream consumers don't mistake all-zeros for a degenerate
        # distribution. The discrete-bins count is sourced from the
        # EnvironmentConfig dataclass default (the trainer doesn't currently
        # override it; the env config dict only sets action_space_type).
        # Probabilities sum to 1.0 exactly when total > 0 (integer counts
        # divided by their sum); when total == 0 (no discrete actions ever
        # taken, e.g. continuous mode) we emit None.
        from src.rl.env import EnvironmentConfig as _EnvCfg
        n_bins = eval_env_config.get(
            'discrete_action_bins',
            _EnvCfg.__dataclass_fields__['discrete_action_bins'].default,
        )
        action_distribution: Optional[Dict[int, float]] = None
        if action_space_type == 'discrete' and episode_results:
            total_action_counts = {i: 0 for i in range(n_bins)}
            for ep in episode_results:
                for k, v in ep.get('action_histogram', {}).items():
                    total_action_counts[int(k)] += int(v)
            total = sum(total_action_counts.values())
            if total > 0:
                action_distribution = {
                    i: total_action_counts[i] / total for i in range(n_bins)
                }
                dist_str = ", ".join(
                    f"bin_{i}={action_distribution[i]:.3f}" for i in range(n_bins)
                )
                logger.info(f"oos_action_distribution fold={fold} {dist_str}")
            else:
                logger.warning(
                    f"oos_action_distribution fold={fold}: zero discrete actions "
                    f"recorded across {len(episode_results)} episodes"
                )
        else:
            logger.info(
                f"oos_action_distribution fold={fold}: skipped "
                f"(action_space_type={action_space_type!r}, histogram is non-meaningful "
                f"for continuous; n_episodes={len(episode_results)})"
            )

        logger.info(f"\n{'='*70}")
        logger.info(f"TEST RESULTS FOR FOLD {fold} (True PnL)")
        logger.info(f"{'='*70}")
        logger.info(f"  Episodes evaluated: {episodes_evaluated}")
        logger.info(f"  Total PnL:  ${total_test_pnl:,.2f}")
        logger.info(f"  Mean PnL:   ${mean_episode_pnl:,.2f}")
        if pnls:
            logger.info(f"  Best:       ${max(pnls):,.2f}")
            logger.info(f"  Worst:      ${min(pnls):,.2f}")
            logger.info(f"  Win Rate:   {winning}/{episodes_evaluated} ({winning/max(episodes_evaluated,1)*100:.0f}%)")

        if episodes_evaluated == 0:
            logger.error("CRITICAL: Zero episodes evaluated - check test data!")
        elif abs(total_test_pnl) < 0.01 and episodes_evaluated > 0:
            logger.warning("Test PnL is ~$0 across all episodes")

        logger.info(f"{'='*70}\n")

        # Save checkpoint
        checkpoint_dir = Path(self.config.output_dir) / f"fold_{fold}_checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        algo.save_checkpoint(str(checkpoint_dir))
        logger.info(f"Checkpoint saved to {checkpoint_dir}")

        phase_at_end = final_result.get('phase', 'unknown')

        result_data = {
            'fold': fold,
            'train_start': train_start.isoformat(),
            'train_end': train_end.isoformat(),
            'test_start': test_start.isoformat(),
            'test_end': test_end.isoformat(),
            'timesteps_total': final_result.get('timesteps_total', 0),
            'phase_at_end': phase_at_end,
            'test_pnl_total': total_test_pnl,
            'test_pnl_mean': mean_episode_pnl,
            'test_episodes_evaluated': episodes_evaluated,
            'test_win_rate': winning / max(episodes_evaluated, 1),
            'per_episode_results': episode_results,
            'best_checkpoint_iteration': best_checkpoint_iteration,
            'best_checkpoint_timestep': best_checkpoint_timestep,
            'best_eval_reward': best_eval_reward if best_eval_reward != float('-inf') else None,
            'validation_checks': checks,
            # Phase 1.2: per-fold OOS action distribution (None for continuous)
            'action_distribution': action_distribution,
            'action_space_type': action_space_type,
        }

        algo.stop()

        return result_data
    
    def _validate_training_mechanics(
        self,
        results: list,
        phase_history: list,
        alpha_history: list,
    ) -> Dict[str, bool]:
        """
        Validate that key training mechanics actually activated.

        Returns dict of check_name -> passed (bool).
        Logs warnings for failures but does NOT raise — this is a smoke test,
        not a hard gate. The user decides whether to proceed.
        """
        checks = {}
        total_timesteps = self.config.warmup_timesteps + self.config.finetune_timesteps

        # CHECK 1: Training reached target timesteps
        final_ts = results[-1].get("timesteps_total", 0) if results else 0
        reached_target = final_ts >= total_timesteps * 0.95
        checks["reached_target_timesteps"] = reached_target
        if reached_target:
            logger.info(f"  ✅ Reached {final_ts}/{total_timesteps} timesteps")
        else:
            logger.warning(f"  ❌ Only reached {final_ts}/{total_timesteps} timesteps")

        # CHECK 2: Joint training mode active
        joint_training = "joint_training" in phase_history
        checks["phase2_reached"] = joint_training
        if joint_training:
            logger.info(f"  ✅ Joint training mode active")
        else:
            logger.warning(f"  ❌ Joint training mode not detected. Phases: {set(phase_history)}")

        # CHECK 3: Alpha schedule decayed (inverted: high → low)
        if alpha_history:
            first_alpha = alpha_history[0][1]
            last_alpha = alpha_history[-1][1]
            alpha_correct = first_alpha > last_alpha and first_alpha >= 0.1
            checks["alpha_ramped"] = alpha_correct
            if alpha_correct:
                logger.info(f"  ✅ Alpha decayed: {first_alpha:.4f} → {last_alpha:.4f}")
            else:
                logger.warning(
                    f"  ❌ Alpha schedule unexpected: {first_alpha:.4f} → {last_alpha:.4f} "
                    f"(expected high→low decay)"
                )
        else:
            checks["alpha_ramped"] = False
            logger.warning("  ❌ No alpha values recorded in results")

        # CHECK 4: Actor LR warmup ramped (low early, high late)
        actor_lrs = [r.get("actor_lr") for r in results if r.get("actor_lr") is not None]
        if actor_lrs:
            had_low_lr = any(lr < 1e-4 for lr in actor_lrs[:min(5, len(actor_lrs))])
            had_full_lr = any(lr >= 1e-4 for lr in actor_lrs[-min(5, len(actor_lrs)):])
            lr_ramped = had_low_lr and had_full_lr
            checks["actor_lr_changed"] = lr_ramped
            if lr_ramped:
                logger.info(f"  ✅ Actor LR warmup: {actor_lrs[0]:.1e} → {actor_lrs[-1]:.1e}")
            else:
                logger.warning(f"  ❌ Actor LR warmup unexpected: {actor_lrs[0]:.1e} → {actor_lrs[-1]:.1e}")
        else:
            checks["actor_lr_changed"] = False
            logger.warning("  ❌ No actor_lr values in results")

        # CHECK 5: Sufficient training iterations
        min_expected_iterations = 15
        enough_iters = len(results) >= min_expected_iterations
        checks["enough_iterations"] = enough_iters
        if enough_iters:
            logger.info(f"  ✅ {len(results)} training iterations completed")
        else:
            logger.warning(
                f"  ❌ Only {len(results)} iterations (expected >= {min_expected_iterations})"
            )

        # SUMMARY
        passed = sum(1 for v in checks.values() if v)
        total = len(checks)
        if passed == total:
            logger.info(f"\n  Pipeline validation: {passed}/{total} checks passed ✅")
        else:
            failed = [k for k, v in checks.items() if not v]
            logger.warning(
                f"\n  Pipeline validation: {passed}/{total} checks passed. "
                f"Failed: {failed}"
            )

        return checks

    def _get_test_setups(self, test_start: str, test_end: str, max_episodes: int = 50) -> list:
        """
        Build validated test setups for a given date range.

        Returns a list of {'symbol': str, 'date': str} dicts for episodes
        that actually load successfully. Used by BOTH the RL evaluation
        and baseline benchmarks to ensure identical test sets.

        Filtering strategy (priority order):
          1. CSV-validated setups (proven profitable in backtests)
          2. High-VWAP parquet setups (max_vwap_dev >= 15%) — lowered from 30%
             to capture more tradeable parabolic setups while filtering out
             episodes with no shorting opportunity.
        """
        from src.rl.data_provider_hybrid import HybridDataProvider
        from src.rl.config import RL_CONFIG
        import random as stdlib_random

        dp = HybridDataProvider(date_range=(test_start, test_end), mode="eval")
        rng = stdlib_random.Random(42)

        # Use the configured VWAP entry threshold as the minimum for eval episodes
        min_vwap_for_eval = RL_CONFIG.get('min_vwap_deviation_entry', 15.0)

        # Gather candidates: CSV first (validated), then VWAP-filtered parquet
        candidates = []
        seen = set()
        for setup in dp.csv_setups:
            key = (setup['symbol'], setup['date'])
            if key not in seen:
                seen.add(key)
                candidates.append(setup)

        parquet_sorted = sorted(
            dp.parquet_setups,
            key=lambda s: s.get('max_vwap_dev', 0),
            reverse=True
        )
        for setup in parquet_sorted:
            key = (setup['symbol'], setup['date'])
            # Filter: only include episodes where VWAP deviation exceeds entry threshold
            if key not in seen and setup.get('max_vwap_dev', 0) >= min_vwap_for_eval:
                seen.add(key)
                candidates.append(setup)

        logger.info(
            f"  Eval candidates: {len(candidates)} "
            f"(CSV: {len([c for c in candidates if c in dp.csv_setups])}, "
            f"Parquet VWAP>={min_vwap_for_eval}%: "
            f"{len(candidates) - len([c for c in candidates if c in dp.csv_setups])})"
        )

        # Shuffle and try to load up to max_episodes (expanded candidate pool)
        rng.shuffle(candidates)
        test_setups = []
        attempts = min(len(candidates), max_episodes * 3)  # Try 3x candidates for max_episodes
        for setup in candidates[:attempts]:
            s, d = setup['symbol'], setup['date']
            df = dp._load_trading_day(s, d)
            if df is not None:
                test_setups.append({'symbol': s, 'date': d})
                if len(test_setups) >= max_episodes:
                    break

        logger.info(f"  Eval setups loaded: {len(test_setups)}/{max_episodes} target")
        return test_setups

    def _run_benchmarks(self, all_results: list) -> Dict[str, Any]:
        """
        Run statistical benchmarks against naive baselines.

        Evaluates NaiveShort and Random agents on the SAME test setups
        as the RL agent, computes bootstrap CIs, and runs permutation tests.
        All numbers are TRUE dollar PnL (not reward signal).
        """
        logger.info(f"\n{'='*70}")
        logger.info("STATISTICAL BENCHMARKS (True PnL)")
        logger.info(f"{'='*70}")

        # RL per-episode PnL from the manual evaluation loop
        rl_pnls = []
        for r in all_results:
            rl_pnls.extend([ep['pnl'] for ep in r.get('per_episode_results', [])])

        benchmark_results = {}
        for r in all_results:
            fold = r['fold']
            test_start = r['test_start'][:10] if isinstance(r['test_start'], str) else r['test_start'].strftime('%Y-%m-%d')
            test_end = r['test_end'][:10] if isinstance(r['test_end'], str) else r['test_end'].strftime('%Y-%m-%d')

            # Baselines do NOT write to trades.jsonl — only RL eval trades go there.
            # This prevents dashboard eval PnL from mixing RL + baseline trades.
            baseline_env_config = {
                "initial_capital": 100000.0,
                "date_range": (test_start, test_end),
                "mode": "eval",
            }

            try:
                # Use the SAME test setups as the RL evaluation
                test_setups = self._get_test_setups(
                    test_start, test_end,
                    max_episodes=self.config.eval_episodes,
                )
                logger.info(f"  Fold {fold}: {len(test_setups)} test episodes")

                # Run Naive Short
                naive_agent = NaiveShortAgent(entry_threshold=15.0)
                naive_metrics = evaluate_baseline_on_fold(
                    agent=naive_agent, test_setups=test_setups,
                    env_config=baseline_env_config, verbose=False
                )
                naive_pnls = [ep['pnl'] for ep in naive_metrics.get('per_episode_results', [])]

                # Run Random (single seed for speed)
                random_agent = RandomAgent(seed=42)
                random_metrics = evaluate_baseline_on_fold(
                    agent=random_agent, test_setups=test_setups,
                    env_config=baseline_env_config, verbose=False
                )
                random_pnls = [ep['pnl'] for ep in random_metrics.get('per_episode_results', [])]

                # True PnL numbers
                naive_total = naive_metrics.get('total_test_pnl', 0) or 0
                random_total = random_metrics.get('total_test_pnl', 0) or 0
                rl_total = r['test_pnl_total']

                # Bootstrap CI on RL per-episode PnL
                fold_rl_pnls = [ep['pnl'] for ep in r.get('per_episode_results', [])]
                rl_ci = bootstrap_confidence_interval(fold_rl_pnls, n_bootstrap=5000) if fold_rl_pnls else {'ci_lower': None, 'ci_upper': None, 'mean': 0}

                logger.info(f"  RL Agent:      ${rl_total:>10,.2f}")
                if rl_ci['ci_lower'] is not None:
                    logger.info(
                        f"    95% CI: [${rl_ci['ci_lower']:,.2f}, ${rl_ci['ci_upper']:,.2f}]"
                    )
                logger.info(f"  Naive Short:   ${naive_total:>10,.2f} ({len(naive_pnls)} episodes)")
                logger.info(f"  Random Agent:  ${random_total:>10,.2f} ({len(random_pnls)} episodes)")

                benchmark_results[f'fold_{fold}'] = {
                    'rl_total': rl_total,
                    'rl_episodes': r.get('test_episodes_evaluated', 0),
                    'naive_short_total': naive_total,
                    'random_total': random_total,
                    'naive_short_episodes': len(naive_pnls),
                    'random_episodes': len(random_pnls),
                    'rl_bootstrap_ci': rl_ci,
                }

            except Exception as e:
                logger.warning(f"  Benchmark error for fold {fold}: {e}")
                benchmark_results[f'fold_{fold}'] = {'error': str(e)}

        logger.info(f"{'='*70}")
        return benchmark_results
    
    def run(self):
        """Run quick test."""

        # Clear dashboard files from previous runs
        metrics_path = Path(self.config.output_dir) / "training_metrics.jsonl"
        trades_path = Path(self.config.output_dir) / "trades.jsonl"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text("")
        trades_path.write_text("")

        # Auto-detect available data range (no hardcoded dates)
        data_min, data_max, total_setups = _detect_data_range()
        available_days = (data_max - data_min).days
        available_months = available_days / 30.0

        # Compute how many months we need for the requested config
        purge_months = self.config.purge_days / 30.0
        months_per_fold = self.config.train_months + purge_months + self.config.test_months
        total_months_needed = months_per_fold + (self.config.n_folds - 1) * self.config.test_months

        logger.info(f"\nAvailable data: {data_min.date()} → {data_max.date()} "
                     f"({available_months:.0f} months, {total_setups} setups)")
        logger.info(f"Requested: train={self.config.train_months}mo + test={self.config.test_months}mo "
                     f"+ purge={self.config.purge_days}d = {months_per_fold:.1f} months/fold")

        if total_months_needed > available_months:
            max_train = int(available_months - self.config.test_months - purge_months)
            max_test = int(available_months - self.config.train_months - purge_months)
            logger.error(
                f"ERROR: Need {total_months_needed:.0f} months but only "
                f"{available_months:.0f} months of data available!\n"
                f"  Options:\n"
                f"    - Max train_months for {self.config.test_months}mo test: {max(0, max_train)}\n"
                f"    - Max test_months for {self.config.train_months}mo train: {max(0, max_test)}\n"
                f"  Reduce train_months or test_months and retry."
            )
            sys.exit(2)  # Exit code 2 = param validation failure (distinct from crash=1)

        # Anchor the window: end at data_max, start as late as possible
        # This ensures the test window uses the most recent data
        end_date = data_max
        start_date = end_date - timedelta(days=30 * total_months_needed)
        start_date = max(start_date, data_min)

        logger.info(f"Computed window: {start_date.date()} → {end_date.date()}")

        splitter = WalkForwardSplitter(
            start_date=start_date,
            end_date=end_date,
            train_years=0,  # We'll override with months
            test_months=self.config.test_months,
            purge_days=self.config.purge_days,
            step_months=self.config.test_months
        )

        # Build fold splits from computed window
        splits = []
        current_start = start_date

        for i in range(self.config.n_folds):
            train_start = current_start
            train_end = train_start + timedelta(days=30 * self.config.train_months)

            purge_start = train_end
            purge_end = purge_start + timedelta(days=self.config.purge_days)

            test_start = purge_end
            test_end = test_start + timedelta(days=30 * self.config.test_months)

            if test_end > end_date:
                logger.warning(
                    f"Fold {i+1} test window ({test_start.date()} → {test_end.date()}) "
                    f"exceeds data end ({end_date.date()}). Skipping."
                )
                break

            logger.info(f"  Fold {len(splits)+1}: Train {train_start.date()} → {train_end.date()} | "
                         f"Test {test_start.date()} → {test_end.date()}")

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

        if not splits:
            logger.error(
                f"No valid folds could be generated! "
                f"train_months={self.config.train_months}, test_months={self.config.test_months} "
                f"doesn't fit in {start_date.date()} → {end_date.date()}. "
                f"Reduce train_months or test_months."
            )
            sys.exit(2)
        
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
        
        avg_pnl = np.mean([r['test_pnl_total'] for r in all_results])
        total_eval_episodes = sum(r.get('test_episodes_evaluated', 0) for r in all_results)

        logger.info(f"Average Test PnL: ${avg_pnl:,.2f} ({total_eval_episodes} episodes)")

        if total_eval_episodes == 0:
            logger.error("CRITICAL: Zero evaluation episodes - check test data!")
        elif abs(avg_pnl) < 0.01 and total_eval_episodes > 0:
            logger.warning("Test PnL is ~$0 - agent may not be learning effectively")
        elif avg_pnl > 0:
            logger.info(f"Agent is profitable on unseen data - ready for full training!")
        else:
            logger.info(f"Agent is negative on unseen data - consider tuning before full run")

        # Aggregate pipeline validation across folds
        all_checks = {}
        for r in all_results:
            for k, v in r.get('validation_checks', {}).items():
                all_checks.setdefault(k, []).append(v)

        if all_checks:
            all_passed = all(all(v) for v in all_checks.values())
            if all_passed:
                logger.info(f"✅ PIPELINE VALIDATION: All checks passed across all folds")
            else:
                failed = {k: v for k, v in all_checks.items() if not all(v)}
                logger.warning(f"❌ PIPELINE VALIDATION FAILURES: {failed}")

        # Statistical benchmarks: compare RL against naive baselines
        try:
            benchmark_data = self._run_benchmarks(all_results)
        except Exception as e:
            logger.error(f"Benchmark error: {e}", exc_info=True)
            benchmark_data = {'error': str(e)}

        # Phase 1.2: aggregate OOS action distribution across ALL folds for
        # the top-level JSON key. Each fold already aggregated its own
        # per-episode histograms; we just merge the per-episode results
        # again (re-summing fold totals would require either storing fold
        # counts or a weighted average; re-summing from episode-level
        # histograms is the simpler invariant — one path, one bug surface).
        action_space_type = getattr(self.config, '_action_space_type', 'continuous')
        top_action_distribution: Optional[Dict[int, float]] = None
        if action_space_type == 'discrete' and all_results:
            # Discover n_bins from the first fold that recorded a non-None
            # distribution; fall back to any per-episode histogram key.
            n_bins = None
            for r in all_results:
                if r.get('action_distribution'):
                    n_bins = len(r['action_distribution'])
                    break
            if n_bins is None:
                for r in all_results:
                    for ep in r.get('per_episode_results', []):
                        hist = ep.get('action_histogram', {})
                        if hist:
                            n_bins = len(hist)
                            break
                    if n_bins is not None:
                        break

            if n_bins:
                total_counts = {i: 0 for i in range(n_bins)}
                for r in all_results:
                    for ep in r.get('per_episode_results', []):
                        for k, v in ep.get('action_histogram', {}).items():
                            total_counts[int(k)] += int(v)
                total = sum(total_counts.values())
                if total > 0:
                    top_action_distribution = {
                        i: total_counts[i] / total for i in range(n_bins)
                    }
                    dist_str = ", ".join(
                        f"bin_{i}={top_action_distribution[i]:.3f}"
                        for i in range(n_bins)
                    )
                    logger.info(f"oos_action_distribution (all folds): {dist_str}")

        # Save results
        results_path = Path(self.config.output_dir) / "quick_test_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'config': {
                    'train_months': self.config.train_months,
                    'test_months': self.config.test_months,
                    'warmup_timesteps': self.config.warmup_timesteps,
                    'finetune_timesteps': self.config.finetune_timesteps,
                    'action_space_type': action_space_type,
                },
                'folds': all_results,
                'aggregate': {
                    'avg_test_pnl': avg_pnl,
                    'total_eval_episodes': total_eval_episodes,
                },
                # Phase 1.2: top-level aggregated OOS action distribution
                # (probabilities sum to 1.0 modulo float rounding). None when
                # action_space_type != 'discrete' OR when no discrete actions
                # were recorded — see per-fold 'action_distribution' for the
                # fold-level breakdown.
                'action_distribution': top_action_distribution,
                'benchmarks': benchmark_data,
            }, f, indent=2, default=str)
        
        logger.info(f"\nResults saved to {results_path}")
        
        ray.shutdown()
        
        return all_results


def main():
    """Main entry point for quick test."""

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

    parser = argparse.ArgumentParser(description='Quick WFO Test (1-2 hours)')
    parser.add_argument('--output-dir', type=str, default='models/wfo_test')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (torch + cuda + numpy + stdlib random) for reproducibility.')
    # Joint training params
    parser.add_argument('--total-steps', type=int, default=25000)
    parser.add_argument('--actor-warmup-steps', type=int, default=2000)
    # Backwards compat: --warmup-steps + --finetune-steps maps to --total-steps
    parser.add_argument('--warmup-steps', type=int, default=None,
                        help='DEPRECATED: mapped to total-steps = warmup + finetune')
    parser.add_argument('--finetune-steps', type=int, default=None,
                        help='DEPRECATED: mapped to total-steps = warmup + finetune')
    parser.add_argument('--train-months', type=int, default=6)
    parser.add_argument('--test-months', type=int, default=1)
    parser.add_argument('--purge-days', type=int, default=5)
    parser.add_argument('--n-folds', type=int, default=1,
                        help='Number of walk-forward folds (default: 1)')
    # SAC hyperparams
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--buffer-size', type=int, default=50000)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--alpha-start', type=float, default=0.2)
    parser.add_argument('--alpha-end', type=float, default=0.01)
    parser.add_argument('--lr-actor', type=float, default=3e-4)
    parser.add_argument('--lr-critic', type=float, default=3e-4)
    # Environment params
    parser.add_argument('--initial-capital', type=float, default=100000.0)
    parser.add_argument('--max-drawdown', type=float, default=-15000.0)
    parser.add_argument('--stop-loss', type=float, default=-2000.0)
    parser.add_argument('--max-pos-fraction', type=float, default=0.30)
    parser.add_argument('--vwap-threshold', type=float, default=15.0)
    parser.add_argument('--txn-cost', type=float, default=0.003)
    parser.add_argument(
        "--r-multiple-reward-weight", type=float, default=0.0,
        help="Per-trade R-multiple reward term weight (0.0 = disabled, matches pre-batch behavior).",
    )
    parser.add_argument(
        "--r-multiple-reward-clip", type=float, default=5.0,
        help="Per-trade R-multiple clip magnitude before scaling.",
    )
    parser.add_argument(
        '--mfe-evap-penalty', type=float, default=0.0,
        help='MFE-evaporation per-step penalty max magnitude (0.0 disables; '
             'try 0.5 to roughly cancel hold_discipline at full evaporation).',
    )
    parser.add_argument(
        '--hold-band-threshold', type=float, default=0.05,
        help='HOLD-band half-width for action discretization. Default 0.05 '
             '(pre-existing). Wider (e.g. 0.3) suppresses noise-driven '
             'micro-covers from the Gaussian policy.',
    )
    parser.add_argument(
        '--entry-threshold', type=float, default=None,
        help='ENTRY threshold override. Default None = falls back to '
             '--hold-band-threshold. Use 0.05 with --cover-threshold 0.3 '
             'for asymmetric thresholds (narrow entry, wide cover).',
    )
    parser.add_argument(
        '--cover-threshold', type=float, default=None,
        help='COVER threshold override. Default None = falls back to '
             '--hold-band-threshold. See --entry-threshold.',
    )
    parser.add_argument('--eval-episodes', type=int, default=50,
                        help='Number of evaluation episodes (default: 50)')
    parser.add_argument(
        '--algo', type=str, default='sac', choices=['sac', 'ppo'],
        help='RL algorithm: sac (default, off-policy + masked Gaussian + '
             'JointTrainingCallback) or ppo (on-policy, no BC warm-start, '
             'no actor-freeze callbacks). PPO is an apples-to-apples algorithm '
             'swap holding the action space constant — see '
             'docs/rl_investigation_synthesis_2026-05-19.md.',
    )
    parser.add_argument(
        '--action-space', type=str, default='continuous',
        choices=['continuous', 'discrete'],
        help='Action space: continuous (default, Box(-1,1)) or discrete '
             '(Discrete(7), bypasses _discretize_action). Designed for use '
             'with --algo ppo. See docs/ppo_continuous_smoke_2026-05-19.md '
             'for the motivation.',
    )

    args = parser.parse_args()

    # Combination guard: SAC has a masked Gaussian policy on Box; it does not
    # support the Discrete action space. Fail fast rather than letting RLlib
    # crash deep in the model wiring.
    if args.algo == 'sac' and args.action_space == 'discrete':
        parser.error("--algo sac is incompatible with --action-space discrete "
                     "(SAC's masked Gaussian model is built for Box actions). "
                     "Use --algo ppo for discrete experiments.")

    # Reproducibility: seed every RNG that matters.
    import random as _stdlib_random
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    _stdlib_random.seed(args.seed)

    # Backwards compat: map old warmup+finetune to total
    if args.warmup_steps is not None and args.finetune_steps is not None:
        total = args.warmup_steps + args.finetune_steps
    else:
        total = args.total_steps

    config = QuickTestConfig(
        output_dir=args.output_dir,
        total_timesteps=total,
        actor_warmup_steps=args.actor_warmup_steps,
        train_months=args.train_months,
        test_months=args.test_months,
        purge_days=args.purge_days,
        n_folds=args.n_folds,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        tau=args.tau,
        gamma=args.gamma,
        alpha_start=args.alpha_start,
        alpha_end=args.alpha_end,
        eval_episodes=args.eval_episodes,
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
    config._mfe_evap_penalty = args.mfe_evap_penalty
    config._hold_band_threshold = args.hold_band_threshold
    config._entry_threshold = args.entry_threshold
    config._cover_threshold = args.cover_threshold
    config._algo = args.algo
    config._action_space_type = args.action_space

    trainer = QuickWFOTrainer(config)
    results = trainer.run()

    return results


if __name__ == "__main__":
    main()
