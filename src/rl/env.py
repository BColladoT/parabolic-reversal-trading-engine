"""
Module III: Custom Reinforcement Learning Environment & Rule-Based Safety Guardrails

PRODUCTION VERSION - Integrated with TCN-AE Perception Module

This module implements a Gymnasium-compatible trading environment for the
Parabolic Reversal Trading strategy with integrated TCN-AE encoding.

Key Features:
1. TCN-AE encoded state representation (64-dim latent + 10 explicit features = 74-dim)
2. True Sortino-based reward with quadratic drawdown penalty
3. Hard action masking enforcement with catastrophic penalty
4. Quarter-Kelly position sizing constraint
5. Circuit breaker at -$19,180 (V5 Relaxed max drawdown)

Author: AI Agent
Date: 2026-03-13
"""

import gymnasium as gym
import numpy as np
import polars as pl
import torch
import random
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
import logging

from src.rl.perception import StateRepresentation, TemporalAutoencoder, PerceptionConfig
from src.rl.data_provider_hybrid import HybridDataProvider, get_data_provider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def r_multiple_reward_term(
    realized_r: float,
    weight: float,
    clip: float = 5.0,
) -> float:
    """Per-trade R-multiple reward contribution (pure-Python, torch-free).

    Returns ``0.0`` when ``weight`` is ``0.0`` (backward-compatible no-op so
    the existing Sortino + drawdown reward path is mathematically unchanged
    for callers that don't opt in). Otherwise clips ``realized_r`` to
    ``[-clip, +clip]`` and multiplies by ``weight``. NaN/inf realized R
    (rare but possible if initial risk is zero) collapses to ``0.0`` so it
    cannot pollute the reward signal.

    Args:
        realized_r: Closed-trade R-multiple (pnl / initial_risk). Can be
            negative; can be NaN/inf if upstream math degenerates.
        weight: Scaling factor. ``0.0`` disables the term entirely.
        clip: Saturation magnitude applied before scaling.

    Returns:
        ``weight * max(-clip, min(clip, realized_r))`` — or ``0.0`` for NaN.
    """
    import math as _math  # local import so helper stays self-contained
    if weight == 0.0:
        return 0.0
    if not _math.isfinite(realized_r):
        return 0.0
    clamped = max(-clip, min(clip, float(realized_r)))
    return float(weight) * clamped


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    timestamp: datetime          # Exit time
    pnl: float                   # Realized PnL after slippage
    win: bool                    # pnl > 0
    return_pct: float            # PnL as % of initial capital
    # Enriched fields for dashboard trade log
    symbol: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    entry_time: object = None    # Optional[datetime] — use object to avoid import issues
    shares: float = 0.0
    bars_held: int = 0
    vwap_at_entry: float = 0.0
    mfe: float = 0.0             # Max favorable excursion (best unrealized PnL)
    mae: float = 0.0             # Max adverse excursion (worst unrealized PnL)


@dataclass
class EnvironmentConfig:
    """Configuration parameters for the trading environment."""
    # Circuit breaker threshold (15% of $100K account — loosened for learning)
    max_single_trade_loss: float = -15000.0
    max_drawdown: float = -15000.0
    circuit_breaker_threshold: float = -15000.0
    
    # Quarter-Kelly constraints
    kelly_lookback_days: int = 30
    kelly_fraction: float = 0.25
    max_leverage_cap: float = 3.0
    min_leverage_floor: float = 0.5
    
    # VWAP threshold (lowered from 20% to 15% for more entry opportunities)
    min_vwap_deviation_entry: float = 15.0
    
    # Position sizing
    max_shares_per_position: int = 5000
    max_position_value: float = 30000.0
    max_position_capital_fraction: float = 0.30  # Position ≤ 30% of current capital
    intra_step_stop_loss: float = -2000.0  # Force-close if unrealized PnL drops below this
    
    # Trading hours (ET)
    entry_window_start: str = "09:45"
    entry_window_end: str = "14:30"
    flatten_time: str = "15:25"
    
    # Sortino calculation
    sortino_lookback: int = 30
    risk_free_rate: float = 0.0
    
    # Reward function parameters
    # Neural normalization: All components scaled to [-10, +10] for SAC stability
    max_acceptable_drawdown: float = -5000.0  # MDD_max ($5k) - early onset for short-only
    circuit_breaker_drawdown: float = -15000.0  # Circuit breaker ($15k — loosened for learning)
    transaction_cost_per_dollar: float = 0.003  # 30 bps (0.30%) realistic micro-cap slippage
    masking_penalty: float = -0.5  # Kept for config compat; no longer applied in step()
    reward_scale: float = 1.0  # Curriculum multiplier applied after percentile scaler
    annealer_total_timesteps: int = 70000  # Total timesteps for RewardAnnealer schedule
    
    # Normalization scales (dollar values → neural range [-10, +10])
    nn_max_penalty: float = -10.0
    nn_max_reward: float = 10.0
    max_pnl_reference: float = 20000.0  # $20k PnL → +10.0 reward

    # R-multiple reward term (A4): opt-in per-trade attribution signal.
    # Default weight 0.0 → mathematically identical to pre-A4 reward.
    r_multiple_reward_weight: float = 0.0
    r_multiple_reward_clip: float = 5.0

    # Action space
    action_space_low: float = -1.0
    action_space_high: float = 1.0


class PercentileRewardScaler:
    """DEPRECATED: Normalizes rewards using running percentiles.
    Replaced by RunningNormScaler due to warmup explosion and signal inversion bugs."""
    def __init__(self, buffer_size=10000, low_pct=5, high_pct=95):
        self.buffer = deque(maxlen=buffer_size)
        self.low_pct = low_pct
        self.high_pct = high_pct

    def scale(self, reward):
        self.buffer.append(reward)
        if len(self.buffer) < 100:
            return reward * 10.0
        low = np.percentile(list(self.buffer), self.low_pct)
        high = np.percentile(list(self.buffer), self.high_pct)
        if high - low < 1e-8:
            return 0.0
        scaled = 10.0 * (reward - low) / (high - low) - 5.0
        return float(np.clip(scaled, -10.0, 10.0))


class RunningNormScaler:
    """Mean/std reward normalization with warmup clipping.
    Fixes PercentileRewardScaler's warmup explosion and signal inversion."""
    def __init__(self, window=5000, clip=10.0, warmup=50):
        self.rewards = deque(maxlen=window)
        self.clip = clip
        self.warmup = warmup

    def scale(self, reward):
        self.rewards.append(reward)
        if len(self.rewards) < self.warmup:
            return float(np.clip(reward, -self.clip, self.clip))
        mean = np.mean(self.rewards)
        std = max(np.std(self.rewards), 1e-6)
        normalized = (reward - mean) / std
        return float(np.clip(normalized, -self.clip, self.clip))


class RewardAnnealer:
    """Cosine-anneals shaping weight from 1.0 to 0.0 over training."""
    def __init__(self, total_timesteps=70000, anneal_start_frac=0.3, anneal_end_frac=0.85):
        self.total = total_timesteps
        self.start = int(total_timesteps * anneal_start_frac)
        self.end = int(total_timesteps * anneal_end_frac)

    def get_shaping_weight(self, timestep):
        if timestep < self.start:
            return 1.0
        elif timestep > self.end:
            return 0.0
        else:
            progress = (timestep - self.start) / (self.end - self.start)
            return 0.5 * (1.0 + np.cos(np.pi * progress))


class ParabolicReversalEnv(gym.Env):
    """
    Production-grade trading environment with TCN-AE perception integration.
    
    State Space (74-dim):
    - [0:64]: TCN-AE latent encoding of 60-bar OHLCV sequence
    - [64]: VWAP deviation (%)
    - [65]: Volume concentration
    - [66]: Current position (normalized)
    - [67]: Unrealized PnL (% of capital)
    - [68]: Current drawdown (%)
    - [69]: Kelly fraction
    - [70:74]: Time features (hour, minute, day_of_week, is_entry_window)
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        config: Optional[Any] = None,
        initial_capital: float = 100000.0,
        render_mode: Optional[str] = None,
        data_provider: Optional[HybridDataProvider] = None,
        perception_model: Optional[StateRepresentation] = None
    ):
        """
        Initialize the trading environment with TCN-AE perception.
        
        Args:
            config: Environment configuration (can include 'date_range' and 'seed')
            initial_capital: Starting portfolio value
            render_mode: Visualization mode
            data_provider: Historical data provider
            perception_model: Pre-trained TCN-AE encoder (frozen)
        """
        super().__init__()
        
        # Handle RLlib EnvContext vs direct EnvConfig
        if config is None:
            self.config = EnvironmentConfig()
            self.date_range = None
            self.env_seed = None
            self.mode = "train"
        elif isinstance(config, EnvironmentConfig):
            self.config = config
            self.date_range = None
            self.env_seed = None
            self.mode = "train"
        elif hasattr(config, 'get'):
            env_context = dict(config)
            self.config = EnvironmentConfig()
            for key in ['initial_capital', 'max_drawdown', 'circuit_breaker_threshold',
                        'reward_scale', 'max_acceptable_drawdown', 'annealer_total_timesteps',
                        'intra_step_stop_loss', 'max_position_capital_fraction',
                        'min_vwap_deviation_entry', 'transaction_cost_per_dollar']:
                if key in env_context:
                    setattr(self.config, key, env_context[key])
            initial_capital = env_context.get('initial_capital', initial_capital)
            # Extract WFO-critical parameters
            self.date_range = env_context.get('date_range', None)
            self.env_seed = env_context.get('seed', None)
            self.mode = env_context.get('mode', 'train')  # CRITICAL: forward mode to provider
        else:
            self.config = EnvironmentConfig()
            self.date_range = None
            self.env_seed = None
            self.mode = "train"
            
        self.initial_capital = initial_capital
        self.render_mode = render_mode

        # Dashboard trade log (set by training scripts via env_config)
        _cfg = dict(config) if hasattr(config, 'get') else {}
        self._trades_log_path = _cfg.get("trades_log_path")
        self._dashboard_fold = _cfg.get("dashboard_fold", 0)
        # Phase tagging: "evaluation" for eval mode, else updated by WarmupCallback
        self._training_phase = "evaluation" if self.mode == "eval" else "unknown"
        if self._trades_log_path:
            logger.info(f"Trade log path: {self._trades_log_path}")
        
        # Data provider - create with date range filter if specified
        if data_provider is not None:
            self.data_provider = data_provider
        else:
            # Pass date constraints AND mode to prevent WFO data leakage
            provider_kwargs = {'mode': self.mode}  # CRITICAL: pass mode (train/eval)
            if self.date_range is not None:
                provider_kwargs['date_range'] = self.date_range
            if self.env_seed is not None:
                provider_kwargs['seed'] = self.env_seed
            self.data_provider = get_data_provider(**provider_kwargs)
        
        # TCN-AE perception model (frozen)
        self.perception_model = perception_model
        if self.perception_model is None:
            # Initialize default perception model
            self._init_default_perception()
        
        # Portfolio state
        # CRITICAL: Separate cash from equity for correct short accounting
        self.cash = initial_capital  # Cash balance (changes with trade proceeds/costs)
        self.current_capital = initial_capital  # Equity (cash + position value, mark-to-market)
        self.prev_capital = initial_capital  # Previous step equity for reward calculation
        self.peak_capital = initial_capital  # Peak equity for drawdown
        self.current_drawdown = 0.0
        self.circuit_breaker_triggered = False
        
        # Position state
        self.current_position = 0.0  # Signed shares: negative for short
        self.current_position_value = 0.0  # Mark-to-market dollar value (negative for short)
        self.entry_price = 0.0  # Weighted average entry price
        self.entry_time = None
        self.unrealized_pnl = 0.0  # Unrealized PnL (position * (price - entry))
        self.realized_pnl_session = 0.0  # Cumulative realized PnL for session
        
        # Trade history for Kelly calculation
        self.trade_history: deque = deque(maxlen=1000)
        self.rolling_kelly_fraction = self.config.min_leverage_floor
        
        # Market state
        self.current_price = 0.0
        self.vwap = 0.0
        self.vwap_deviation = 0.0
        self.volume_concentration = 0.0
        self.current_time = None
        self.in_entry_window = False
        self.must_flatten = False
        
        # Episode tracking
        self.episode_pnl = 0.0
        self.episode_trades = 0
        self.episode_wins = 0
        self.daily_returns: deque = deque(maxlen=self.config.sortino_lookback)
        
        # 60-bar history buffer for TCN-AE encoding
        self.price_history: deque = deque(maxlen=60)

        # Reward shaping infrastructure (Changes 4 & 8)
        self.reward_scaler = RunningNormScaler()
        self.reward_annealer = RewardAnnealer(total_timesteps=self.config.annealer_total_timesteps)
        self.global_timestep = 0
        self.prev_drawdown = 0.0

        # Trade lifecycle tracking (Change 6)
        self.max_favorable_excursion = 0.0  # Best unrealized PnL during trade
        self.max_adverse_excursion = 0.0    # Worst unrealized PnL during trade
        self.bars_in_trade = 0
        self.bars_since_last_trade = 0
        self._entry_cooldown = 0            # Bars remaining on entry bonus cooldown
        self._just_entered = False          # True on the step a position opens
        # Trade entry snapshots (captured at OPEN_SHORT, read at _record_trade)
        self._trade_entry_price = 0.0
        self._trade_entry_time = None
        self._trade_shares = 0.0
        self._trade_entry_vwap = 0.0
        
        # Define action space: continuous [-1, 1]
        self.action_space = gym.spaces.Box(
            low=self.config.action_space_low,
            high=self.config.action_space_high,
            shape=(1,),
            dtype=np.float32
        )
        
        # Observation space: 74-dim state + 3-dim action mask + 1-dim kelly
        self.observation_space = gym.spaces.Dict({
            'state': gym.spaces.Box(-np.inf, np.inf, (74,), dtype=np.float32),
            'action_mask': gym.spaces.Box(0, 1, (3,), dtype=np.int8),
            'kelly_leverage': gym.spaces.Box(0, 5, (1,), dtype=np.float32)
        })
        
        logger.info(f"Environment initialized with capital: ${initial_capital:,.2f}")
        logger.info(f"Circuit breaker threshold: ${self.config.circuit_breaker_threshold:,.2f}")
    
    def _init_default_perception(self):
        """Initialize default TCN-AE perception model."""
        perception_cfg = PerceptionConfig()
        autoencoder = TemporalAutoencoder(perception_cfg)
        autoencoder.freeze_encoder()
        self.perception_model = StateRepresentation(autoencoder.encoder, perception_cfg)
        logger.info("Initialized default TCN-AE perception model")
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        # Set Python random seed
        random.seed(seed)
        # Set numpy seed
        np.random.seed(seed)
        # Set torch seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Propagate to data provider if it has a resettable RNG
        if hasattr(self.data_provider, '_rng'):
            self.data_provider._rng = random.Random(seed)
            self.data_provider.seed = seed
        logger.debug(f"Environment seed set to {seed}")

    def update_curriculum_params(self, reward_scale: float, max_drawdown: float) -> None:
        """
        Update curriculum-controlled parameters mid-training without recreating the env.

        Called by WarmupCallback when CurriculumManager signals a phase transition.

        Args:
            reward_scale: New reward multiplier (applied to raw_reward before percentile scaler).
            max_drawdown: New circuit-breaker threshold (negative dollar amount, e.g. -15000).
        """
        self.config.reward_scale = reward_scale
        self.config.max_drawdown = max_drawdown
        logger.info(
            f"Curriculum params updated: reward_scale={reward_scale}, "
            f"max_drawdown={max_drawdown}"
        )

    def _load_specific_episode(self, symbol: str, date_str: str) -> bool:
        """
        Load a specific episode by symbol and date without random sampling.
        
        CRITICAL: This method is used for deterministic evaluation to ensure
        that exactly the requested (symbol, date) episode is loaded, without
        any random resampling.
        
        Args:
            symbol: Stock symbol
            date_str: Date string in YYYY-MM-DD format
            
        Returns:
            True if episode loaded successfully, False otherwise
        """
        try:
            # Load the trading day data directly
            df = self.data_provider._load_trading_day(symbol, date_str)
            if df is None:
                return False
            
            # Find first bar where VWAP > 20% (entry threshold - 3% buffer)
            from src.rl.config import RL_CONFIG
            entry_threshold = RL_CONFIG.get('min_vwap_deviation_entry', 20.0)
            valid_bars = df.filter(pl.col('vwap_dev').abs() > (entry_threshold - 3.0))
            
            if len(valid_bars) == 0:
                logger.warning(f"No valid entry bars for {symbol} {date_str}")
                return False
            
            # Get starting index
            first_valid_bar = valid_bars.row(0, named=True)
            if '__row_index__' in first_valid_bar:
                start_bar_idx = int(first_valid_bar['__row_index__'])
            else:
                # Find index by filtering
                first_ts = first_valid_bar['timestamp']
                all_timestamps = df['timestamp'].to_list()
                start_bar_idx = all_timestamps.index(first_ts)
            
            # Set provider state directly (bypassing random reset_episode)
            self.data_provider.current_data = df
            self.data_provider.current_symbol = symbol
            self.data_provider.current_date = date_str
            self.data_provider.start_bar_idx = start_bar_idx
            self.data_provider.current_bar_idx = start_bar_idx
            
            logger.info(f"[FIXED] Loaded specific episode: {symbol} {date_str} "
                       f"(bar {start_bar_idx}/{len(df)})")
            return True
            
        except Exception as e:
            logger.error(f"Error loading specific episode {symbol} {date_str}: {e}")
            return False
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducible episode selection
            options: Optional dictionary with reset options:
                - "fixed_setup": {"symbol": str, "date": str} - Load specific episode
                  When provided, skips random episode sampling for deterministic evaluation.
        
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)

        # Log episode-end diagnostics summary (before counters are cleared)
        if hasattr(self, '_mask_violation_count') and self._mask_violation_count > 0:
            total = self._mask_violation_count
            expected = getattr(self, '_expected_violations', 0)
            suspicious = getattr(self, '_suspicious_violations', 0)
            source = getattr(self, '_episode_source', '?')
            max_vwap = getattr(self, '_episode_max_vwap_dev', 0.0)

            summary = (
                f"Episode violations: {total} "
                f"(expected={expected}, suspicious={suspicious}) | "
                f"source={source}, max_vwap_dev={max_vwap:.1f}%"
            )
            if suspicious > 0:
                logger.warning(f"SUSPICIOUS: {summary}")
            else:
                logger.info(summary)

            self._mask_violation_count = 0
        if hasattr(self, '_high_shaping_count') and self._high_shaping_count > 0:
            logger.info(f"Episode high-shaping steps: {self._high_shaping_count}")
            self._high_shaping_count = 0

        # Set seed for reproducible episode selection
        if seed is not None:
            self._set_seed(seed)

        # Reset portfolio
        self.cash = self.initial_capital
        self.current_capital = self.initial_capital
        self.prev_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.current_drawdown = 0.0
        self.circuit_breaker_triggered = False
        
        # Reset position
        self.current_position = 0.0
        self.current_position_value = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.unrealized_pnl = 0.0
        self.realized_pnl_session = 0.0
        
        # Reset tracking
        self.episode_pnl = 0.0
        self.episode_trades = 0
        self.episode_wins = 0
        self.daily_returns.clear()
        self.price_history.clear()
        self.trade_history.clear()

        # Reset violation classification counters
        self._expected_violations = 0
        self._suspicious_violations = 0
        self._episode_max_vwap_dev = 0.0
        self._episode_source = 'unknown'

        # Reset trade lifecycle / reward shaping state
        self.max_favorable_excursion = 0.0
        self.max_adverse_excursion = 0.0
        self.bars_in_trade = 0
        self.bars_since_last_trade = 0
        self._entry_cooldown = 0
        self._just_entered = False
        self.prev_drawdown = 0.0
        
        # Check for fixed setup option (deterministic evaluation)
        fixed_setup = options.get("fixed_setup") if options else None
        
        episode_load_failed = False
        
        if fixed_setup is not None:
            # CRITICAL: Load specific episode without random sampling
            symbol = fixed_setup["symbol"]
            date_str = fixed_setup["date"]
            success = self._load_specific_episode(symbol, date_str)
            if not success:
                logger.warning(f"Failed to load fixed episode: {symbol} {date_str}")
                # Clear ALL stale state to prevent invariant violations
                self.data_provider.current_data = None
                self.data_provider.current_symbol = None
                self.data_provider.current_date = None
                self.data_provider.start_bar_idx = 0
                self.data_provider.current_bar_idx = 0
                # Clear env market state for neutral observation
                self.current_price = 0.0
                self.vwap = 0.0
                self.vwap_deviation = 0.0
                self.volume_concentration = 0.0
                self.current_time = None
                self.in_entry_window = False
                self.must_flatten = False
                # Clear price history for neutral observation
                self.price_history.clear()
                # Return neutral observation with failure flag
                observation = self._get_observation()
                info = {"episode_load_failed": True}
                return observation, info
        else:
            # Standard random episode selection
            success = self.data_provider.reset_episode()
            if not success:
                # Training mode: retry with different episodes
                max_retries = 5
                for retry in range(max_retries):
                    logger.warning(f"Failed to load trading day data, retry {retry+1}/{max_retries}")
                    success = self.data_provider.reset_episode()
                    if success:
                        break
                
                if not success:
                    raise RuntimeError(
                        f"Failed to load any valid training episode after {max_retries} retries. "
                        f"CSV setups: {len(self.data_provider.csv_setups)}, "
                        f"Parquet setups: {len(self.data_provider.parquet_setups)}"
                    )
        
        # Pull episode metadata from data provider for violation classification
        if self.data_provider:
            self._episode_source = getattr(self.data_provider, 'current_source', 'unknown')

        # Initialize with first bar
        bar = self.data_provider.get_current_bar()
        if bar:
            self._load_market_state({
                'price': bar.close,
                'vwap': bar.vwap,
                'vwap_deviation': bar.vwap_deviation,
                'volume_concentration': 0.0,
                'timestamp': bar.timestamp
            })
            # Initialize price history with STRICTLY PRE-DECISION bars
            # CRITICAL: Window is [current_bar_idx - 60, current_bar_idx)
            # Current bar is EXCLUDED - no future leakage
            ohlcv_history = self.data_provider.get_pre_decision_sequence(lookback=60)
            if ohlcv_history is not None and ohlcv_history.shape == (60, 5):
                for i in range(60):
                    self.price_history.append({
                        'open': float(ohlcv_history[i, 0]),
                        'high': float(ohlcv_history[i, 1]),
                        'low': float(ohlcv_history[i, 2]),
                        'close': float(ohlcv_history[i, 3]),
                        'volume': float(ohlcv_history[i, 4])
                    })
                logger.debug(f"Loaded 60 pre-decision bars at reset (window ends at bar {self.data_provider.current_bar_idx - 1})")
            else:
                logger.warning(f"Could not load pre-decision bars at reset: shape={ohlcv_history.shape if ohlcv_history is not None else None}")
        
        self._update_kelly_fraction()
        
        observation = self._get_observation()
        info = self._get_info()
        info["episode_load_failed"] = False
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """
        Execute one timestep with correct chronological MDP order:
        1. Process action (record target position at time t)
        2. Advance to next bar (price changes to P_{t+1})
        3. Calculate reward using NEW price P_{t+1}
        4. Check termination
        5. Return observation
        """
        # Store previous equity and position for reward calculation
        self.prev_capital = self.current_capital
        prev_position = self.current_position

        # Validate action shape
        action = np.clip(action, self.config.action_space_low, self.config.action_space_high)
        desired_exposure_fraction = float(action[0])
        
        # CANONICAL ACTION CONVENTION (must match agent.py):
        #   action < -0.05: INCREASE short exposure (Entry/Add)
        #   action > 0.05:  DECREASE short exposure (Cover)
        #   else:           HOLD current exposure  [-0.05, 0.05]
        # Asymmetric thresholds break HOLD attractor; zero-mean Gaussian → COVER
        assert -1.0 <= desired_exposure_fraction <= 1.0, \
            f"Action {desired_exposure_fraction} out of bounds [-1, 1]"
        
        # Get current action mask BEFORE we move
        action_mask = self._compute_action_mask()
        
        # === CIRCUIT BREAKER CHECK ===
# === CIRCUIT BREAKER CHECK ===
        if self.circuit_breaker_triggered:
            logger.warning("CIRCUIT BREAKER ACTIVE - Forcing position close")
            if self.current_position != 0:
                self._close_position()
            raw_reward = self._calculate_true_reward() + self._compute_drawdown_penalty()
            reward = self.reward_scaler.scale(raw_reward)
            observation = self._get_observation()
            info = self._get_info()
            return observation, reward, True, False, info
        
        # === A. PROCESS ACTION at time t ===
        # Determine action type from continuous value
        action_type = self._discretize_action(desired_exposure_fraction)
        
        # Neural masking (Change 10) should prevent invalid actions before they reach the env.
        # RLlib legacy API samples from the distribution externally, so soft violations are
        # possible early in training. Track and clamp rather than crash.
        if action_mask[action_type] == 0:
            self._mask_violation_count = getattr(self, "_mask_violation_count", 0) + 1

            # Classify: expected (VWAP too low or outside window) vs suspicious
            if action_type == 0:  # Entry attempt blocked
                if self.vwap_deviation < self.config.min_vwap_deviation_entry:
                    self._expected_violations += 1
                elif not self.in_entry_window:
                    self._expected_violations += 1
                else:
                    self._suspicious_violations += 1
            else:
                self._expected_violations += 1

            # Log first 3 in detail, then periodic summaries with breakdown
            if self._mask_violation_count <= 3:
                logger.warning(
                    f"MASK VIOLATION #{self._mask_violation_count}: action_type={action_type} "
                    f"overridden to HOLD. mask={action_mask}, "
                    f"VWAP dev={self.vwap_deviation:.2f}, position={self.current_position}"
                )
            elif self._mask_violation_count % 1000 == 0:
                logger.info(
                    f"Mask violations so far: {self._mask_violation_count} "
                    f"(expected={self._expected_violations}, suspicious={self._suspicious_violations})"
                )
            desired_exposure_fraction = 0.0
            action_type = 2  # Hold
        
        # === HOLD SEMANTICS: True no-op on position ===
        # When action_type is HOLD (2), preserve current position exactly.
        # This prevents accidental flattening from small action values in the hold band.
        if action_type == 2:  # HOLD
            # Preserve current position - no transaction costs for true hold
            target_position_value = self.current_position_value
        else:
            # Apply Quarter-Kelly position sizing for non-HOLD actions
            max_leverage = self._calculate_kelly_constrained_leverage()
            target_exposure = desired_exposure_fraction * max_leverage
            
            # Convert to dollar value and apply hard caps
            target_position_value = target_exposure * self.current_capital
            
            # STRICT SHORT-ONLY CAP: Never allow a positive dollar position (Long)
            target_position_value = min(0.0, target_position_value)
            
            # Capital-relative position cap: position ≤ fraction of current capital
            # Prevents outsized bets when account is depleted after losses
            capital_cap = self.config.max_position_capital_fraction * max(self.current_capital, 0.0)
            effective_max = min(self.config.max_position_value, capital_cap)

            # Stop-loss-bounded position cap: size the position so that
            # a realistic adverse move won't exceed the intra-step stop loss.
            # This is the PRIMARY defense against catastrophic gap losses on penny stocks.
            # Sub-$3 micro-caps can gap 300-500%+ in squeezes (e.g. PEPG $2→$12).
            stop_loss_limit = abs(self.config.intra_step_stop_loss)
            vwap_dev = abs(self.vwap_deviation)
            if vwap_dev > 80:
                est_adverse = 0.50
            elif vwap_dev > 50:
                est_adverse = 0.40
            elif vwap_dev > 30:
                est_adverse = 0.30
            else:
                est_adverse = 0.20
            price = max(self.current_price, 0.01)
            if price < 3.0:
                # Sub-$3: assume 10x worst-case gap (e.g. SEV $4→$34)
                est_adverse = max(est_adverse, 10.0)
            elif price < 5.0:
                # $3-$5: assume 5x worst-case gap
                est_adverse = max(est_adverse, 5.0)
            elif price < 10.0:
                # $5-$10: assume 3x worst-case gap
                est_adverse = max(est_adverse, 3.0)
            elif price < 30.0:
                # $10-$30: assume 1x worst-case gap (100% move)
                est_adverse = max(est_adverse, 1.0)
            else:
                # $30+: assume 50% worst-case gap
                est_adverse = max(est_adverse, 0.50)
            if est_adverse > 0:
                stop_loss_cap = stop_loss_limit / est_adverse
                effective_max = min(effective_max, stop_loss_cap)

            target_position_value = np.clip(
                target_position_value,
                -effective_max,
                0.0  # Upper bound is strictly 0.0 (no long positions)
            )
        
        # Check pre-trade loss projection
        potential_loss = self._estimate_potential_loss(target_position_value)
        if potential_loss < self.config.max_single_trade_loss:
            logger.warning(
                f"TRADE BLOCKED: Potential loss ${potential_loss:,.2f} "
                f"exceeds limit ${self.config.max_single_trade_loss:,.2f}"
            )
            target_position_value = 0.0
        
        # Execute position change (sets target position)
        executed = self._execute_position_change(target_position_value)
        
        # === B. SAVE CURRENT BAR AND ADVANCE (TIME MOVES FORWARD: t -> t+1) ===
        # CONVENTION: price_history stores the EXACT pre-decision sequence [t-60, t)
        # where t = current_bar_idx. The current bar is NOT in price_history.
        # 
        # After advancing to t+1, we need price_history = [t+1-60, t+1) = [t-59, t+1)
        # which is [t-59, t] in bar indices (excluding t+1).
        #
        # We have [t-60, t) before advance. We need to add bar t.
        # So we save the current bar BEFORE advancing.
        current_bar_for_history = self.data_provider.get_current_bar()
        
        # Now advance to next bar
        next_bar = self.data_provider.advance()

        # Check if we've reached end of day
        done = self.data_provider.is_done()

        # ALWAYS append the bar we just left, even if next_bar is None.
        # This preserves the invariant that price_history stores the exact
        # pre-decision sequence ending at current_bar_idx - 1.
        if current_bar_for_history is not None:
            self.price_history.append({
                "open": float(current_bar_for_history.open),
                "high": float(current_bar_for_history.high),
                "low": float(current_bar_for_history.low),
                "close": float(current_bar_for_history.close),
                "volume": float(current_bar_for_history.volume),
            })

            assert len(self.price_history) <= 60, (
                f"price_history exceeded maxlen: {len(self.price_history)}"
            )

        # Only update market state if a real next bar exists.
        if next_bar:
            new_price = float(next_bar.close)
            self._load_market_state({
                "price": new_price,
                "vwap": float(next_bar.vwap),
                "vwap_deviation": float(next_bar.vwap_deviation),
                "volume_concentration": 0.0,
                "timestamp": next_bar.timestamp,
            })

            # Update unrealized PnL using the new bar price
            if self.current_position != 0 and self.entry_price != 0:
                self.unrealized_pnl = self.current_position * (new_price - self.entry_price)

                # INTRA-STEP STOP LOSS: Force-close if unrealized loss exceeds limit.
                # This is the primary defense against catastrophic single-bar gap losses.
                # Without this, a position opened at bar t survives an unlimited price
                # gap at bar t+1 with no exit mechanism.
                if self.unrealized_pnl < self.config.intra_step_stop_loss:
                    logger.warning(
                        f"INTRA-STEP STOP: Unrealized PnL ${self.unrealized_pnl:,.2f} "
                        f"< limit ${self.config.intra_step_stop_loss:,.2f}. "
                        f"Force-closing position."
                    )
                    self._close_position()

        # Save drawdown before update (used for recovery bonus in Change 7)
        self.prev_drawdown = self.current_drawdown

        # Update portfolio metrics with NEW price
        self._update_portfolio_metrics()
        
        # Check circuit breaker — ACTIVE: immediately flatten position
        if self.current_drawdown <= self.config.max_drawdown:
            self.circuit_breaker_triggered = True
            logger.critical(
                f"CIRCUIT BREAKER TRIGGERED: Drawdown ${self.current_drawdown:,.2f}. "
                f"Force-closing position."
            )
            if self.current_position != 0:
                self._close_position()
                self._update_portfolio_metrics()
        
        # === C. CALCULATE COMPOSITE REWARD using NEW price P_{t+1} ===
        self.global_timestep += 1

        # Update trade lifecycle tracking BEFORE computing shaping bonuses
        self._just_entered = (prev_position == 0.0 and self.current_position < 0.0)
        if self._entry_cooldown > 0:
            self._entry_cooldown -= 1
        prev_bars_in_trade = self.bars_in_trade
        if self.current_position < 0:
            self.bars_in_trade += 1
            self.bars_since_last_trade = 0
            self.max_favorable_excursion = max(self.max_favorable_excursion, self.unrealized_pnl)
            self.max_adverse_excursion = min(self.max_adverse_excursion, self.unrealized_pnl)
        else:
            self.bars_in_trade = 0
            self.bars_since_last_trade += 1

        base_reward = self._calculate_true_reward()
        drawdown_penalty = self._compute_drawdown_penalty()

        # Shaping components (annealed to zero as training matures)
        w = self.reward_annealer.get_shaping_weight(self.global_timestep)
        opportunity_cost = self._compute_opportunity_cost()
        participation = self._compute_participation_bonus()
        waiting_quality = self._compute_waiting_quality()
        entry_bonus = self._compute_entry_bonus()
        hold_discipline = self._compute_hold_discipline()

        completion_bonus = 0.0
        r_multiple_term = 0.0  # A4: per-trade R-multiple reward (default weight 0 = no-op)
        if prev_position < 0 and self.current_position == 0:
            trade_pnl = getattr(self, '_last_trade_pnl', 0.0)
            completion_bonus = self._compute_trade_completion_bonus(
                trade_pnl=trade_pnl,
                bars_held=prev_bars_in_trade,
                mfe=self.max_favorable_excursion,
                mae=self.max_adverse_excursion,
            )
            # A4: realized R-multiple = trade_pnl / per-trade risk denominator.
            # Use abs(max_acceptable_drawdown) as the R unit ($5K default — the
            # per-trade risk threshold already configured for drawdown shaping).
            risk_denom = abs(self.config.max_acceptable_drawdown)
            realized_r = trade_pnl / risk_denom if risk_denom > 1e-6 else 0.0
            r_multiple_term = r_multiple_reward_term(
                realized_r=realized_r,
                weight=self.config.r_multiple_reward_weight,
                clip=self.config.r_multiple_reward_clip,
            )
            # Reset excursion trackers after trade close
            self.max_favorable_excursion = 0.0
            self.max_adverse_excursion = 0.0

        shaping_total = (opportunity_cost + participation + waiting_quality
                         + entry_bonus + hold_discipline + completion_bonus)

        # Hard-cap shaping at 3x base reward magnitude to prevent shaping dominance.
        # When base is near-zero (flat, no equity change), allow small absolute shaping.
        max_shaping_abs = max(3.0 * abs(base_reward), 0.5)  # floor of 0.5 for flat periods
        if abs(shaping_total) > max_shaping_abs:
            shaping_total = np.sign(shaping_total) * max_shaping_abs

        # A4: r_multiple_term is added outside the shaping cap because it is an
        # attribution signal (not shaping). Defaults to 0.0 when the opt-in
        # weight is 0.0, preserving pre-A4 reward exactly.
        raw_reward = base_reward + drawdown_penalty + (w * shaping_total) + r_multiple_term
        # Curriculum reward scale applied AFTER percentile normalization (FIX 1)
        reward = self.reward_scaler.scale(raw_reward) * self.config.reward_scale

        # === D. CHECK TERMINATION ===
        terminated = self.circuit_breaker_triggered or done
        truncated = False

        # Record unclosed positions as trades at episode end (for dashboard)
        # Record unclosed positions as trades at episode end.
        # Guard: _trade_entry_price > 0 ensures we only record if there's a
        # real unclosed trade (cleared after COVER_SHORT, so no double-counting).
        if (terminated and abs(self.current_position) > 0.001
                and self._trade_entry_price > 0):
            self._record_trade(self.unrealized_pnl)

        # === E. PREPARE AND RETURN OBSERVATION ===
        self._update_kelly_fraction()
        observation = self._get_observation()
        info = self._get_info()

        # FIX 10: shaping_ratio monitoring
        shaping_ratio = shaping_total / (abs(base_reward) + 1e-8)
        info['shaping_ratio'] = shaping_ratio

        # Only flag shaping dominance when base_reward is material (equity actually changed).
        # When base ≈ 0 (agent flat, no equity delta), high ratio is a math artifact, not bonus farming.
        if abs(base_reward) > 1e-4 and abs(shaping_total) > 5.0 * abs(base_reward):
            self._high_shaping_count = getattr(self, '_high_shaping_count', 0) + 1
            # Log first occurrence, then every 1000th to avoid flooding
            if self._high_shaping_count == 1 or (
                self._high_shaping_count % 1000 == 0
            ):
                logger.warning(
                    f"High shaping ratio (#{self._high_shaping_count}): {shaping_ratio:.2f} "
                    f"(base={base_reward:.4f}, shaping={shaping_total:.4f})"
                )

        return observation, reward, terminated, truncated, info
    
    def _calculate_true_reward(self) -> float:
        """
        Base reward: normalized incremental equity change.

        INVARIANT: sum(base_rewards) = (final_equity - initial) / initial * 1000
        Drawdown penalty and shaping are computed separately and combined in step().
        """
        equity_delta = self.current_capital - self.prev_capital
        return float((equity_delta / self.initial_capital) * 1000.0)

    # ------------------------------------------------------------------
    # Change 7: Softened drawdown penalty with recovery bonus
    # ------------------------------------------------------------------
    def _compute_drawdown_penalty(self) -> float:
        """
        Softened drawdown penalty replacing original quadratic.
        - Linear onset:  $0–$2K  →  0.0
        - Linear zone:   $2K–$5K → [-1.0, 0.0]
        - Quadratic:     $5K–$10K → [-10.0, -1.0]  (was -50.0 max)
        - Recovery bonus: up to +0.5 per step for reducing drawdown.
        """
        current_dd = abs(self.current_drawdown)
        prev_dd = abs(self.prev_drawdown)

        if current_dd <= 2000.0:
            dd_penalty = 0.0
        elif current_dd <= 5000.0:
            dd_penalty = -1.0 * (current_dd - 2000.0) / 3000.0
        elif current_dd <= 10000.0:
            excess = (current_dd - 5000.0) / 5000.0
            dd_penalty = -1.0 - 9.0 * (excess ** 2)
        else:
            dd_penalty = -10.0

        recovery_bonus = 0.0
        if current_dd < prev_dd and prev_dd > 0:
            recovery_ratio = (prev_dd - current_dd) / prev_dd
            recovery_bonus = min(0.5, recovery_ratio * 5.0)

        return dd_penalty + recovery_bonus

    def _compute_unrealized_loss_penalty(self) -> float:
        """Per-trade unrealized loss penalty (permanent, not annealed).
        $0-$200 grace, $200-$500 linear, $500+ quadratic. Capped at -10.0."""
        if self.unrealized_pnl >= 0 or self.current_position == 0:
            return 0.0
        loss = abs(self.unrealized_pnl)
        if loss <= 200:
            return 0.0
        if loss <= 500:
            return -0.5 * (loss - 200) / 300
        excess = (loss - 500) / 500
        return -0.5 - 2.0 * min(excess ** 2, 4.75)

    # ------------------------------------------------------------------
    # Change 5: Opportunity cost, participation, strategic waiting
    # ------------------------------------------------------------------
    def _compute_opportunity_cost(self) -> float:
        """Penalize HOLD when flat during valid entry window."""
        if self.current_position != 0:
            return 0.0
        if not self.in_entry_window:
            return 0.0
        if self.vwap_deviation < self.config.min_vwap_deviation_entry:
            return 0.0
        # Scale penalty with how far above threshold we are
        excess = (self.vwap_deviation - self.config.min_vwap_deviation_entry) / 100.0
        return -0.1 - min(0.2, excess * 0.2)  # base -0.1, max -0.3

    def _compute_participation_bonus(self) -> float:
        """+0.15 per step for holding short while VWAP deviation > 10%."""
        if self.current_position >= 0:
            return 0.0
        return 0.15 if self.vwap_deviation > 10.0 else 0.0

    def _compute_waiting_quality(self) -> float:
        """Reward strategic patience when flat (reduced 4x to avoid HOLD dominance)."""
        if self.current_position != 0:
            return 0.0
        if not self.in_entry_window:
            return 0.005  # Being flat outside entry window (was 0.02)
        if self.vwap_deviation < self.config.min_vwap_deviation_entry:
            return 0.002  # Below entry threshold — wait for setup (was 0.01)
        return 0.0

    # ------------------------------------------------------------------
    # Change 6: Trade lifecycle reward shaping
    # ------------------------------------------------------------------
    def _compute_entry_bonus(self) -> float:
        """One-time bonus on position open, scaled by setup quality."""
        if not self._just_entered:
            return 0.0
        if self._entry_cooldown > 0:
            return 0.0
        vwap_factor = min(1.0, max(0.0,
            (self.vwap_deviation - self.config.min_vwap_deviation_entry) / 80.0))
        vol_factor = float(np.clip(self.volume_concentration, 0.0, 1.0))
        momentum_factor = vwap_factor  # proxy: higher deviation = stronger move
        entry_quality = (vwap_factor + vol_factor + momentum_factor) / 3.0
        bonus = 0.2 + entry_quality * 1.8  # [0.2, 2.0]
        self._entry_cooldown = 5
        return bonus

    def _compute_hold_discipline(self) -> float:
        """Per-step reward for holding a profitable short position."""
        if self.current_position >= 0:
            return 0.0
        if self.unrealized_pnl <= 0:
            return 0.0
        profit_pct = self.unrealized_pnl / max(self.initial_capital, 1.0)
        return 0.1 + min(0.4, profit_pct * 20.0)  # [0.1, 0.5]

    def _compute_trade_completion_bonus(
        self,
        trade_pnl: float,
        bars_held: int,
        mfe: float,
        mae: float,
    ) -> float:
        """One-time bonus on position close."""
        # FIX 2: Capped at 1.5 total (0.5 + 0.5 + 0.5) to prevent bonus farming
        win_bonus = 0.5 if trade_pnl > 0 else 0.0

        if mfe > 0 and trade_pnl > 0:
            capture_ratio = min(1.0, trade_pnl / mfe)
            efficiency_bonus = capture_ratio * 0.5  # max 0.5
        else:
            efficiency_bonus = 0.0

        if mae < 0:
            mae_pct = abs(mae) / max(self.initial_capital, 1.0)
            risk_bonus = min(0.5, max(0.0, 1.0 - mae_pct * 50.0))  # max 0.5
        else:
            risk_bonus = 0.5

        return win_bonus + efficiency_bonus + risk_bonus
    
    def _discretize_action(self, exposure_fraction: float) -> int:
        """
        Discretize continuous action for a STRICTLY SHORT-ONLY strategy.

        CANONICAL ACTION CONVENTION (must match agent.py):
            action in [-1, 1]
            action < -0.05: INCREASE short exposure (Entry/Add)
            action > 0.05:  DECREASE short exposure (Cover)
            else:           HOLD current exposure  [-0.05, 0.05]

        Asymmetric thresholds (§2 Option C): center of a zero-mean Gaussian
        maps to COVER territory, breaking the HOLD attractor.

        Action mask indices:
            mask[0]: increase short allowed
            mask[1]: decrease short/cover allowed
            mask[2]: hold allowed

        Returns:
            0: INCREASE short exposure (Entry) - exposure_fraction < -0.05
            1: DECREASE short exposure (Cover) - exposure_fraction > 0.05
            2: Hold (-0.05 <= exposure_fraction <= 0.05)
        """
        if exposure_fraction < -0.05:
            return 0  # Action 0: INCREASE short exposure (Entry)
        elif exposure_fraction > 0.05:
            return 1  # Action 1: DECREASE short exposure (Cover)
        else:
            return 2  # Action 2: Hold
    
    def _compute_action_mask(self) -> np.ndarray:
        """Compute the boolean action mask."""
        mask = np.ones(3, dtype=np.int8)
        
        # Circuit breaker - only allow closing
        if self.circuit_breaker_triggered:
            mask[0] = 0
            mask[2] = 0
            if self.current_position == 0:
                mask[1] = 0
            return mask
        
        # Time-based restrictions
        if not self.in_entry_window:
            mask[0] = 0  # No new entries outside window
        
        if self.must_flatten:
            mask[0] = 0
            mask[2] = 0 if self.current_position != 0 else 1
        
        # VWAP deviation threshold: must exceed entry threshold
        if self.vwap_deviation < self.config.min_vwap_deviation_entry:
            mask[0] = 0  # No entries below threshold
        
        # Position-based logic
        if self.current_position == 0:
            mask[1] = 0  # Can't decrease if flat
        
        # Maximum position value constraint
        current_exposure = abs(self.current_position_value)
        if current_exposure >= self.config.max_position_value:
            mask[0] = 0  # Block increasing beyond max
        
        return mask
    
    def _calculate_kelly_constrained_leverage(self) -> float:
        """Calculate maximum allowable leverage based on Quarter-Kelly."""
        # During training: use fixed 1.0x leverage so agent learns full sizing
        if getattr(self, 'mode', 'train') == 'train':
            return 1.0

        if len(self.trade_history) < 10:
            return self.config.min_leverage_floor
        
        # Calculate win rate and profit factor from recent trades
        recent_trades = list(self.trade_history)[-self.config.kelly_lookback_days:]
        wins = [t for t in recent_trades if t.win]
        losses = [t for t in recent_trades if not t.win]
        
        if not wins or not losses:
            return self.config.min_leverage_floor
        
        win_rate = len(wins) / len(recent_trades)
        avg_win = np.mean([t.return_pct for t in wins])
        avg_loss = abs(np.mean([t.return_pct for t in losses]))
        
        if avg_loss == 0:
            return self.config.min_leverage_floor
        
        # Kelly fraction: f* = (p(b+1) - 1) / b
        # where b = avg_win / avg_loss
        b = avg_win / avg_loss
        if b <= 0:
            return self.config.min_leverage_floor
        
        kelly_full = (win_rate * (b + 1) - 1) / b
        
        # Quarter-Kelly constraint
        kelly_constrained = max(
            self.config.min_leverage_floor,
            min(kelly_full * self.config.kelly_fraction, self.config.max_leverage_cap)
        )
        
        return kelly_constrained
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Generate the observation dict for the current decision point.

        State semantics:
        - `price_history` stores the exact pre-decision window:
        [current_bar_idx - 60, current_bar_idx)
        - The last real bar in the sequence should therefore correspond to
        raw data index `current_bar_idx - 1`.
        - The current bar itself must NOT be included in the encoded sequence.
        - Missing prefix history is padded with zeros only.

        Returns:
            {
                "state": np.ndarray shape (74,),
                "action_mask": np.ndarray shape (3,),
                "kelly_leverage": np.ndarray shape (1,)
            }
        """
        # ------------------------------------------------------------------
        # 1) Build exact 60-bar pre-decision OHLCV window
        # ------------------------------------------------------------------
        history_list = list(self.price_history)

        if len(history_list) < 60:
            padding_needed = 60 - len(history_list)
            zero_bar = {
                "open": 0.0,
                "high": 0.0,
                "low": 0.0,
                "close": 0.0,
                "volume": 0.0,
            }
            history_list = [zero_bar] * padding_needed + history_list

        pre_decision_bars = history_list[-60:]

        # ------------------------------------------------------------------
        # 2) Structural invariants
        # ------------------------------------------------------------------
        assert len(pre_decision_bars) == 60, (
            f"Sequence length must be exactly 60, got {len(pre_decision_bars)}"
        )

        if __debug__:
            dp = self.data_provider

            # Formal invariant:
            # If we have raw data and are past bar 0, the last sequence bar must
            # match raw bar at current_bar_idx - 1.
            if (
                dp is not None
                and dp.current_data is not None
                and getattr(dp, "current_bar_idx", None) is not None
                and dp.current_bar_idx > 0
                and dp.current_bar_idx - 1 < len(dp.current_data)
            ):
                expected_idx = dp.current_bar_idx - 1
                expected_bar = dp.current_data.row(expected_idx, named=True)

                actual_close = float(pre_decision_bars[-1]["close"])
                expected_close = float(expected_bar["close"])

                if abs(actual_close - expected_close) > 1e-4:
                    raise AssertionError(
                        "Window misalignment detected: "
                        f"sequence last close={actual_close}, "
                        f"expected raw close at index {expected_idx}={expected_close}. "
                        "Expected sequence to end at current_bar_idx - 1."
                    )

            # Heuristic only:
            # Equal closes do NOT prove leakage because consecutive bars can
            # legitimately share the same close.
            current_bar = dp.get_current_bar() if dp is not None else None
            if current_bar is not None:
                last_seq_close = float(pre_decision_bars[-1]["close"])
                current_close = float(current_bar.close)
                if abs(last_seq_close - current_close) < 1e-4:
                    logger.debug(
                        "HEURISTIC: Last sequence bar close equals current bar close "
                        f"({last_seq_close}). This can be harmless because consecutive "
                        "bars may legitimately have the same close."
                    )

        # ------------------------------------------------------------------
        # 3) Convert OHLCV bars to tensor [5, 60]
        # ------------------------------------------------------------------
        ohlcv_tensor = np.zeros((5, 60), dtype=np.float32)
        for i, bar in enumerate(pre_decision_bars):
            ohlcv_tensor[0, i] = float(bar["open"])
            ohlcv_tensor[1, i] = float(bar["high"])
            ohlcv_tensor[2, i] = float(bar["low"])
            ohlcv_tensor[3, i] = float(bar["close"])
            ohlcv_tensor[4, i] = float(bar["volume"])

        # Simple per-feature z-score normalization over the 60-bar window
        for j in range(5):
            mean = np.mean(ohlcv_tensor[j, :])
            std = np.std(ohlcv_tensor[j, :]) + 1e-8
            ohlcv_tensor[j, :] = (ohlcv_tensor[j, :] - mean) / std

        # ------------------------------------------------------------------
        # 4) TCN-AE latent encoding
        # ------------------------------------------------------------------
        with torch.no_grad():
            device = next(self.perception_model.encoder.parameters()).device
            x = torch.FloatTensor(ohlcv_tensor).unsqueeze(0).to(device)  # [1, 5, 60]
            latent = self.perception_model.encoder(x).squeeze(0).cpu().numpy()  # [64]

        # ------------------------------------------------------------------
        # 5) Explicit features (10 dims)
        # ------------------------------------------------------------------
        position_norm = self.current_position / (self.config.max_shares_per_position + 1e-8)
        unrealized_pnl_pct = self.unrealized_pnl / (self.current_capital + 1e-8)
        drawdown_pct = self.current_drawdown / (self.initial_capital + 1e-8)

        if self.current_time:
            hour = self.current_time.hour / 24.0
            minute = self.current_time.minute / 60.0
            day_of_week = self.current_time.weekday() / 7.0
            is_entry_window = 1.0 if self.in_entry_window else 0.0
        else:
            hour = 0.0
            minute = 0.0
            day_of_week = 0.0
            is_entry_window = 0.0

        explicit_features = np.array(
            [
                self.vwap_deviation / 100.0,
                self.volume_concentration,
                position_norm,
                unrealized_pnl_pct,
                drawdown_pct,
                self.rolling_kelly_fraction / self.config.max_leverage_cap,
                hour,
                minute,
                day_of_week,
                is_entry_window,
            ],
            dtype=np.float32,
        )

        # ------------------------------------------------------------------
        # 6) Final observation dict
        # ------------------------------------------------------------------
        state = np.concatenate([latent, explicit_features]).astype(np.float32)
        action_mask = self._compute_action_mask()
        kelly_leverage = np.array([self.rolling_kelly_fraction], dtype=np.float32)

        return {
            "state": state,
            "action_mask": action_mask,
            "kelly_leverage": kelly_leverage,
        }
    
    def _execute_position_change(self, target_value: float) -> bool:
        """
        Execute position change with financially correct share-based accounting.
        
        Key invariant: We track position in SIGNED SHARES, not dollar values.
        Dollar values are computed as shares * price for mark-to-market.
        
        Supports:
        - Opening new short position
        - Adding to existing short (scale-in) with weighted avg entry update
        - Partial cover (scale-out with realized PnL on covered portion only)
        - Full cover (realize remaining PnL)
        
        Position tracking:
        - current_position: signed share count (negative for short)
        - current_position_value: derived as shares * current_price (for UI/masks)
        - entry_price: weighted average entry price (updated on adds, unchanged on covers)
        - realized_pnl_session: cumulative realized PnL (updated on covers)
        """
        prev_value = self.current_position_value
        delta_value = target_value - prev_value
        
        if abs(delta_value) < 1.0:  # Min trade size
            return False
        
        # Calculate slippage/fees on dollar delta
        slippage_cost = abs(delta_value) * self.config.transaction_cost_per_dollar
        
        # Determine trade direction and type
        # For shorts: target_value is negative (e.g., -$1000 = short $1000 worth)
        if prev_value == 0 and target_value < 0:
            # OPEN NEW SHORT
            trade_type = 'OPEN_SHORT'
        elif prev_value < 0 and target_value < prev_value:
            # ADD TO SHORT (more negative = larger position)
            trade_type = 'ADD_SHORT'
        elif prev_value < 0 and target_value > prev_value:
            # COVER SHORT (less negative = smaller position)
            trade_type = 'COVER_SHORT'
        else:
            # Invalid or no-op (e.g., trying to go long)
            return False
        
        if self.current_price <= 0:
            return False
        
        # Convert dollar values to SHARES at current price
        # This is the key: we trade SHARES, dollar exposure changes with price
        target_shares = target_value / self.current_price  # negative for short target
        prev_shares = self.current_position  # negative if short
        delta_shares = target_shares - prev_shares  # negative for add, positive for cover
        
        if trade_type == 'OPEN_SHORT':
            # NEW SHORT: Record entry price and position in shares
            self.entry_price = self.current_price
            self.entry_time = self.current_time
            self.current_position = target_shares  # negative
            self.current_position_value = self.current_position * self.current_price
            # Snapshot for dashboard trade log
            self._trade_entry_price = self.current_price
            self._trade_entry_time = self.current_time
            self._trade_shares = abs(target_shares)
            self._trade_entry_vwap = self.vwap
            
            # Cash: receive proceeds minus fees
            notional = abs(target_shares) * self.current_price
            self.cash += notional - slippage_cost
            # episode_pnl is computed automatically in _update_portfolio_metrics()
            self._last_trade_pnl = -slippage_cost
            
        elif trade_type == 'ADD_SHORT':
            # ADD TO SHORT: Update weighted average entry price
            # Formula: (old_shares * old_avg + new_shares * fill_price) / new_total
            new_total_shares = prev_shares + delta_shares  # more negative
            
            # Weighted average: preserve total cost basis
            # Both terms negative: old_shares * old_avg = negative cost basis
            old_cost_basis = prev_shares * self.entry_price
            new_cost_basis = delta_shares * self.current_price
            self.entry_price = (old_cost_basis + new_cost_basis) / new_total_shares
            
            self.current_position = new_total_shares
            self.current_position_value = self.current_position * self.current_price
            # Update trade snapshot to reflect total position size
            self._trade_shares = abs(new_total_shares)

            # Cash: receive additional proceeds minus fees
            add_notional = abs(delta_shares) * self.current_price
            self.cash += add_notional - slippage_cost
            # episode_pnl is computed automatically in _update_portfolio_metrics()
            self._last_trade_pnl = -slippage_cost

        elif trade_type == 'COVER_SHORT':
            # COVER: Realize PnL on covered portion only
            # shares_to_cover is positive (reducing negative position)
            shares_to_cover = min(abs(delta_shares), abs(prev_shares))
            
            # Realized PnL = shares_covered * (avg_entry - cover_price)
            # For profitable short: entry > cover_price -> positive PnL
            cover_price = self.current_price
            realized_pnl = shares_to_cover * (self.entry_price - cover_price)
            
            # Update cumulative realized PnL
            self.realized_pnl_session += realized_pnl
            
            # Cash: pay to buy back shares plus fees
            cover_cost = shares_to_cover * cover_price
            self.cash -= cover_cost + slippage_cost
            
            # episode_pnl is computed automatically in _update_portfolio_metrics()
            trade_pnl = realized_pnl - slippage_cost
            self._last_trade_pnl = trade_pnl
            
            # Update position: target_shares might be slightly different due to
            # rounding, so use actual calculation
            self.current_position = prev_shares + shares_to_cover  # toward zero
            self.current_position_value = self.current_position * self.current_price
            
            # Record trade for Kelly calculation ONLY when fully covered
            # (round-trip complete). Partial covers are not recorded.
            if abs(self.current_position) < 0.001:
                self._record_trade(trade_pnl)
                self.entry_price = 0.0
                self.entry_time = None
                self.unrealized_pnl = 0.0
        
        return True
    
    def _record_trade(self, pnl: float):
        """Record completed trade with full details for Kelly calc and dashboard."""
        trade = TradeRecord(
            timestamp=self.current_time or datetime.now(),
            pnl=pnl,
            win=pnl > 0,
            return_pct=(pnl / self.initial_capital) * 100,
            symbol=getattr(self.data_provider, 'current_symbol', '') or '',
            entry_price=self._trade_entry_price,
            exit_price=self.current_price,
            entry_time=self._trade_entry_time,
            shares=self._trade_shares,
            bars_held=self.bars_in_trade,
            vwap_at_entry=self._trade_entry_vwap,
            mfe=self.max_favorable_excursion,
            mae=self.max_adverse_excursion,
        )
        self.trade_history.append(trade)

        # Write to dashboard trades log (if path configured)
        trades_path = getattr(self, '_trades_log_path', None)
        if trades_path:
            try:
                import json as _json
                from pathlib import Path as _Path
                # Ensure parent dir exists
                _Path(trades_path).parent.mkdir(parents=True, exist_ok=True)
                ep_date = getattr(self.data_provider, 'current_date', '') or ''
                line = _json.dumps({
                    "fold": int(getattr(self, '_dashboard_fold', 0)),
                    "phase": getattr(self, '_training_phase', 'unknown'),
                    "symbol": str(trade.symbol), "date": str(ep_date),
                    "entry_price": round(float(trade.entry_price), 2),
                    "exit_price": round(float(trade.exit_price), 2),
                    "entry_time": trade.entry_time.isoformat() if hasattr(trade.entry_time, 'isoformat') else str(trade.entry_time or ''),
                    "exit_time": trade.timestamp.isoformat() if hasattr(trade.timestamp, 'isoformat') else str(trade.timestamp or ''),
                    "shares": round(float(trade.shares), 1),
                    "pnl": round(float(trade.pnl), 2),
                    "return_pct": round(float(trade.return_pct), 3),
                    "win": bool(trade.win),
                    "bars_held": int(trade.bars_held),
                    "vwap_at_entry": round(float(trade.vwap_at_entry), 2),
                    "mfe": round(float(trade.mfe), 2),
                    "mae": round(float(trade.mae), 2),
                })
                with open(trades_path, "a") as f:
                    f.write(line + "\n")
            except Exception:
                pass

        self.episode_trades += 1
        if trade.win:
            self.episode_wins += 1

        # Clear trade snapshots to prevent reuse by EOD recording
        self._trade_entry_price = 0.0
        self._trade_entry_time = None
        self._trade_shares = 0.0
        self._trade_entry_vwap = 0.0
        # Reset excursion trackers for next trade
        self.max_favorable_excursion = 0.0
        self.max_adverse_excursion = 0.0

    def _update_portfolio_metrics(self):
        """
        Update equity, drawdown, and returns.
        
        CRITICAL INVARIANTS:
        - Equity = Cash + Position Value (mark-to-market)
        - episode_pnl = Equity - Initial_Capital (total account performance)
        - For shorts: position_value is negative (a liability)
        """
        # Update position value at current price
        if self.current_position != 0 and self.current_price > 0:
            self.current_position_value = self.current_position * self.current_price
            self.unrealized_pnl = self.current_position * (self.current_price - self.entry_price)
        else:
            self.current_position_value = 0.0
            self.unrealized_pnl = 0.0
        
        # Equity = Cash + Position Value (position value is negative for shorts)
        self.current_capital = self.cash + self.current_position_value

        # BANKRUPTCY PROTECTION: Never allow negative equity in the simulation.
        # Negative capital produces nonsensical Kelly fractions and position sizes.
        if self.current_capital < 0.0:
            logger.critical(
                f"BANKRUPTCY: Capital ${self.current_capital:,.2f} < $0. "
                f"Force-closing position, clamping to $0."
            )
            if self.current_position != 0:
                self._close_position()
            self.current_capital = max(0.0, self.cash + self.current_position_value)
            self.circuit_breaker_triggered = True

        # INVARIANT: episode_pnl must equal total account performance
        self.episode_pnl = self.current_capital - self.initial_capital
        
        # Update peak capital
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        # Calculate drawdown
        self.current_drawdown = self.current_capital - self.peak_capital
        
        # Track daily return for Sortino
        if len(self.daily_returns) == 0 or self._is_new_day():
            daily_return = (self.current_capital - self.initial_capital) / self.initial_capital
            self.daily_returns.append(daily_return)
    
    def _update_kelly_fraction(self):
        """Update rolling Kelly fraction."""
        self.rolling_kelly_fraction = self._calculate_kelly_constrained_leverage()
    
    def _close_position(self):
        """Close current position."""
        if self.current_position != 0:
            self._execute_position_change(0.0)
        self.current_position = 0.0
        self.current_position_value = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
    
    def _estimate_potential_loss(self, target_value: float) -> float:
        """Estimate potential loss from position change.

        Uses VWAP-deviation-tiered adverse move estimates reflecting
        actual micro-cap squeeze behavior: highly extended stocks have
        the most gap risk (50-200% squeezes are common).

        For penny stocks (< $5), adverse move multipliers are scaled up
        because sub-$5 stocks routinely gap 200-500% in squeezes.
        """
        vwap_dev = abs(self.vwap_deviation)
        if vwap_dev > 80:
            adverse_move = 0.50  # 50% gap risk at extreme extension
        elif vwap_dev > 50:
            adverse_move = 0.40
        elif vwap_dev > 30:
            adverse_move = 0.30
        else:
            adverse_move = 0.20  # 20% baseline for moderate extension

        # Price-based gap risk: micro-caps can gap 300-1000%+ in squeezes.
        price = max(self.current_price, 0.01)
        if price < 3.0:
            adverse_move = max(adverse_move, 10.0)
        elif price < 5.0:
            adverse_move = max(adverse_move, 5.0)
        elif price < 10.0:
            adverse_move = max(adverse_move, 3.0)
        elif price < 30.0:
            adverse_move = max(adverse_move, 1.0)
        else:
            adverse_move = max(adverse_move, 0.50)

        return -abs(target_value) * adverse_move
    
    def _load_market_state(self, state: Dict):
        """Load market state from external source."""
        self.current_price = state.get('price', 0.0)
        self.vwap = state.get('vwap', 0.0)
        self.vwap_deviation = state.get('vwap_deviation', 0.0)
        self._episode_max_vwap_dev = max(
            getattr(self, '_episode_max_vwap_dev', 0.0),
            abs(self.vwap_deviation)
        )
        self.volume_concentration = state.get('volume_concentration', 0.0)
        self.current_time = state.get('timestamp', datetime.now())
        self._update_time_constraints()
    
    def _update_time_constraints(self):
        """Update time-based constraints."""
        if self.current_time is None:
            return
        
        current_time_str = self.current_time.strftime("%H:%M")
        self.in_entry_window = (
            self.config.entry_window_start <= current_time_str <= self.config.entry_window_end
        )
        self.must_flatten = current_time_str >= self.config.flatten_time
    
    def _is_end_of_trading_day(self) -> bool:
        """Check if trading day has ended."""
        if self.data_provider and self.data_provider.is_done():
            return True
        if self.current_time is not None:
            return self.current_time.strftime("%H:%M") >= "16:00"
        return False
    
    def _is_new_day(self) -> bool:
        """Check if this is a new trading day."""
        return False  # Simplified - would track day changes
    
    def _get_info(self) -> Dict:
        """Get additional information."""
        return {
            'capital': self.current_capital,
            'drawdown': self.current_drawdown,
            'position': self.current_position,
            'trades': self.episode_trades,
            'wins': self.episode_wins,
            'vwap_deviation': self.vwap_deviation,
            'price': self.current_price,
            'vwap': self.vwap,
            'time': self.current_time,
            'episode_pnl': self.episode_pnl,  # True equity PnL (base, no bonuses)
            'episode_max_vwap_dev': getattr(self, '_episode_max_vwap_dev', 0.0),
            'episode_source': getattr(self, '_episode_source', 'unknown'),
            'expected_violations': getattr(self, '_expected_violations', 0),
            'suspicious_violations': getattr(self, '_suspicious_violations', 0),
        }
    
    def render(self):
        """Render environment state."""
        if self.render_mode == "human":
            print(f"Capital: ${self.current_capital:,.2f} | "
                  f"Drawdown: ${self.current_drawdown:,.2f} | "
                  f"Position: ${self.current_position_value:,.2f} | "
                  f"VWAP Dev: {self.vwap_deviation:.2f}% | "
                  f"Kelly: {self.rolling_kelly_fraction:.2f}x")
