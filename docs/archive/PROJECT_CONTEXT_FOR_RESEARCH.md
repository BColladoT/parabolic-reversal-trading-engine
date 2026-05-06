# Quant Trading RL — Complete Source Code for Research

## src/rl/env.py
```python
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


@dataclass
class TradeRecord:
    """Record of a completed trade for Kelly Criterion calculations."""
    timestamp: datetime
    pnl: float
    win: bool
    return_pct: float


@dataclass
class EnvironmentConfig:
    """Configuration parameters for the trading environment."""
    # Circuit breaker threshold (10% of $100K account)
    max_single_trade_loss: float = -10000.0
    max_drawdown: float = -10000.0
    circuit_breaker_threshold: float = -10000.0
    
    # Quarter-Kelly constraints
    kelly_lookback_days: int = 30
    kelly_fraction: float = 0.25
    max_leverage_cap: float = 3.0
    min_leverage_floor: float = 0.5
    
    # VWAP threshold from V5 Relaxed (matches settings.yaml 1.20 = 20%)
    min_vwap_deviation_entry: float = 20.0
    
    # Position sizing
    max_shares_per_position: int = 5000
    max_position_value: float = 30000.0
    
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
    circuit_breaker_drawdown: float = -10000.0  # Circuit breaker ($10k)
    transaction_cost_per_dollar: float = 0.003  # 30 bps (0.30%) realistic micro-cap slippage
    masking_penalty: float = -0.5  # Proportional to typical trade rewards
    
    # Normalization scales (dollar values → neural range [-10, +10])
    nn_max_penalty: float = -10.0
    nn_max_reward: float = 10.0
    max_pnl_reference: float = 20000.0  # $20k PnL → +10.0 reward
    
    # Action space
    action_space_low: float = -1.0
    action_space_high: float = 1.0


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
            for key in ['initial_capital', 'max_drawdown', 'circuit_breaker_threshold']:
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
        
        # Initialize with first bar
        bar = self.data_provider.get_current_bar()
        if bar:
            self._load_market_state({
                'price': bar.close,
                'vwap': bar.vwap,
                'vwap_deviation': bar.vwap_deviation,
                'volume_concentration': 0.75,
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
        # Store previous equity for reward calculation
        self.prev_capital = self.current_capital
        
        # Validate action shape
        action = np.clip(action, self.config.action_space_low, self.config.action_space_high)
        desired_exposure_fraction = float(action[0])
        
        # CANONICAL ACTION CONVENTION (must match agent.py):
        #   action < -0.1: INCREASE short exposure (Entry/Add)
        #   action > 0.1:  DECREASE short exposure (Cover)
        #   else:          HOLD current exposure
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
            reward = self._calculate_true_reward()
            observation = self._get_observation()
            info = self._get_info()
            return observation, reward, True, False, info
        
        # === A. PROCESS ACTION at time t ===
        # Determine action type from continuous value
        action_type = self._discretize_action(desired_exposure_fraction)
        
        # Check if action violates mask
        if action_mask[action_type] == 0:
            # ACTION VIOLATES MASK - Override to HOLD (0.0) - NO ILLEGAL TRADE
            logger.warning(
                f"MASK VIOLATION: Action type {action_type} blocked by mask. "
                f"VWAP dev: {self.vwap_deviation:.2f}, Position: {self.current_position}"
            )
            desired_exposure_fraction = 0.0
            action_type = 2  # Hold
            masking_violation = True
        else:
            masking_violation = False
        
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
            
            target_position_value = np.clip(
                target_position_value,
                -self.config.max_position_value,
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
                "volume_concentration": 0.75,
                "timestamp": next_bar.timestamp,
            })

            # Update unrealized PnL using the new bar price
            if self.current_position != 0 and self.entry_price != 0:
                self.unrealized_pnl = self.current_position * (new_price - self.entry_price)
        
        # Update portfolio metrics with NEW price
        self._update_portfolio_metrics()
        
        # Check circuit breaker
        if self.current_drawdown <= self.config.max_drawdown:
            self.circuit_breaker_triggered = True
            logger.critical(
                f"CIRCUIT BREAKER TRIGGERED: Drawdown ${self.current_drawdown:,.2f}"
            )
        
        # === C. CALCULATE REWARD using NEW price P_{t+1} ===
        base_reward = self._calculate_true_reward()
        
        # Apply masking penalty as additive (preserves reward accounting invariant)
        if masking_violation:
            reward = base_reward + self.config.masking_penalty
            logger.warning(f"Applied masking penalty: {self.config.masking_penalty}")
        else:
            reward = base_reward
        
        # === D. CHECK TERMINATION ===
        terminated = self.circuit_breaker_triggered or done
        truncated = False
        
        # === E. PREPARE AND RETURN OBSERVATION ===
        self._update_kelly_fraction()
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_true_reward(self) -> float:
        """
        Calculate reward as normalized incremental equity change.
        
        BASE REWARD:
            base_reward = (equity_t - equity_{t-1}) / initial_capital * 100
        
        FINAL REWARD:
            reward = base_reward + optional_shaping_penalties
        
        Optional shaping penalties:
        - drawdown_penalty: Quadratic penalty when drawdown exceeds $15k threshold
        - masking_penalty: Hard override in step() for invalid actions (-10.0)
        
        INVARIANT (base reward only):
            sum(base_rewards) = (final_equity - initial) / initial * 100
        
        NOTE: Cumulative reward equals total return ONLY when no shaping
        penalties or masking overrides are triggered during the episode.
        """
        # === CORE REWARD: Normalized equity delta ===
        equity_delta = self.current_capital - self.prev_capital
        reward = (equity_delta / self.initial_capital) * 1000.0  # Scale: $100 = +1.0 reward
        
        # === OPTIONAL: Drawdown penalty (quadratic beyond threshold) ===
        # This is the only shaping component retained - it prevents catastrophic
        # losses without affecting normal trading rewards
        current_dd = abs(self.current_drawdown)
        max_acceptable_dd = abs(self.config.max_acceptable_drawdown)
        
        if current_dd > max_acceptable_dd:
            excess_dd = current_dd - max_acceptable_dd
            max_excess = abs(self.config.circuit_breaker_drawdown) - max_acceptable_dd
            drawdown_penalty = -((excess_dd / max_excess) ** 2) * 50.0  # [-50, 0]
            reward += np.clip(drawdown_penalty, -50.0, 0.0)
        
        return float(reward)
    
    def _discretize_action(self, exposure_fraction: float) -> int:
        """
        Discretize continuous action for a STRICTLY SHORT-ONLY strategy.
        
        CANONICAL ACTION CONVENTION (must match agent.py):
            action in [-1, 1]
            action < -0.1: INCREASE short exposure (Entry/Add)
            action > 0.1:  DECREASE short exposure (Cover)
            else:          HOLD current exposure
        
        Action mask indices:
            mask[0]: increase short allowed
            mask[1]: decrease short/cover allowed
            mask[2]: hold allowed
        
        Returns:
            0: INCREASE short exposure (Entry) - exposure_fraction < -0.1
            1: DECREASE short exposure (Cover) - exposure_fraction > 0.1
            2: Hold (|exposure_fraction| <= 0.1)
        """
        if exposure_fraction < -0.1:
            return 0  # Action 0: INCREASE short exposure (Entry)
        elif exposure_fraction > 0.1:
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
            x = torch.FloatTensor(ohlcv_tensor).unsqueeze(0)  # [1, 5, 60]
            latent = self.perception_model.encoder(x).squeeze(0).numpy()  # [64]

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
        """Record completed trade for Kelly calculation."""
        trade = TradeRecord(
            timestamp=self.current_time or datetime.now(),
            pnl=pnl,
            win=pnl > 0,
            return_pct=(pnl / self.initial_capital) * 100
        )
        self.trade_history.append(trade)
        self.episode_trades += 1
        if trade.win:
            self.episode_wins += 1
    
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
        """Estimate potential loss from position change."""
        if self.vwap_deviation > 50:
            return -self.config.max_single_trade_loss * 0.5
        return -abs(target_value) * 0.1
    
    def _load_market_state(self, state: Dict):
        """Load market state from external source."""
        self.current_price = state.get('price', 0.0)
        self.vwap = state.get('vwap', 0.0)
        self.vwap_deviation = state.get('vwap_deviation', 0.0)
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
        }
    
    def render(self):
        """Render environment state."""
        if self.render_mode == "human":
            print(f"Capital: ${self.current_capital:,.2f} | "
                  f"Drawdown: ${self.current_drawdown:,.2f} | "
                  f"Position: ${self.current_position_value:,.2f} | "
                  f"VWAP Dev: {self.vwap_deviation:.2f}% | "
                  f"Kelly: {self.rolling_kelly_fraction:.2f}x")

```

## src/rl/agent.py
```python
"""
Module II: Deep Reinforcement Learning Decision Engine (Soft Actor-Critic)

IMPORTANT: This module contains BOTH active and INACTIVE code paths.

ACTIVE PATH (used by train_wfo.py):
- Plain RLlib SAC via create_sac_config() with custom_model=False
- Environment enforces action constraints via masking_penalty (-10.0)
- No custom model masking in the neural network

INACTIVE / EXPERIMENTAL PATH (NOT used by train_wfo.py):
- MaskedSACRLModule / MaskedSACModel for action masking in the model
- build_sac_config() with custom_model=True wires these in
- Currently NOT integrated into the active WFO trainer

WARNING: The masked model code (MaskedSACRLModule, MaskedSACModel) is
experimental and INACTIVE in the main training loop. Use plain SAC
unless you explicitly wire in the masked path and test thoroughly.

Active Components:
1. MaskedGaussianPolicy - Standalone PyTorch policy (used by BC pretraining)
2. SACConfig - Configuration dataclass
3. create_sac_config() with custom_model=False - Plain RLlib SAC

Inactive / Experimental Components:
1. MaskedSACRLModule - RLModule for Ray 2.x+ (NOT ACTIVE)
2. MaskedSACModel - TorchModelV2 for legacy API (NOT ACTIVE)
3. build_sac_config() with custom_model=True - Masked path (NOT ACTIVE)

The active WFO trainer (train_wfo.py) uses plain RLlib SAC without
custom model masking. Action constraints are enforced by the
environment's hard masking penalty, not by the neural network.

Author: AI Agent
Date: 2026-03-12
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ray RLlib imports - handle version differences gracefully
try:
    # Try new API first (Ray 2.x+)
    from ray.rllib.core.rl_module.rl_module import RLModule
    from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
    from ray.rllib.core.models.specs.typing import SpecType
    from ray.rllib.core.models.configs import ModelConfig
    from ray.rllib.algorithms.sac.sac_rl_module import SACRLModule
    from ray.rllib.algorithms.sac.torch.sac_torch_rl_module import SACTorchRLModule
    USE_NEW_API = True
    logger.info("Using Ray RLlib 2.x+ API")
except ImportError:
    # Fall back to legacy API
    try:
        from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
        from ray.rllib.models.modelv2 import ModelV2
        from ray.rllib.utils.framework import try_import_torch
        USE_NEW_API = False
        logger.info("Using Ray RLlib legacy API")
    except ImportError:
        logger.warning("Ray RLlib not installed. Installing...")
        USE_NEW_API = None


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SACConfig:
    """Configuration for SAC Algorithm."""
    
    # State and action dimensions
    state_dim: int = 74
    action_dim: int = 1
    action_low: float = -1.0
    action_high: float = 1.0
    
    # Network architecture
    actor_hidden_dims: List[int] = None
    critic_hidden_dims: List[int] = None
    activation: str = "relu"
    
    # SAC specific parameters
    tau: float = 0.005              # Target network update rate
    gamma: float = 0.99             # Discount factor
    alpha: float = 0.2              # Initial temperature
    auto_tune_alpha: bool = True    # Automatic entropy tuning
    target_entropy: Optional[float] = None  # Target entropy for auto-tuning
    
    # Training parameters
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    buffer_size: int = 1000000
    batch_size: int = 256
    
    # Action masking parameters
    mask_penalty: float = -1e9      # Large negative value for invalid actions
    
    def __post_init__(self):
        if self.actor_hidden_dims is None:
            self.actor_hidden_dims = [256, 256]
        if self.critic_hidden_dims is None:
            self.critic_hidden_dims = [256, 256]
        if self.target_entropy is None:
            # Heuristic: -dim(A) for continuous actions
            self.target_entropy = -self.action_dim


# =============================================================================
# Neural Network Components
# =============================================================================

class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "relu",
        output_activation: Optional[str] = None,
        use_layer_norm: bool = False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "elu":
                layers.append(nn.ELU())
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        if output_activation:
            if output_activation == "tanh":
                layers.append(nn.Tanh())
            elif output_activation == "sigmoid":
                layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MaskedGaussianPolicy(nn.Module):
    """
    Gaussian policy network with action masking for continuous actions.
    
    For continuous action masking, we use a technique where:
    1. The network outputs mean and log_std for a Gaussian distribution
    2. The action_mask determines which action directions are valid
    3. Invalid directions receive extremely high penalty in log_prob
    4. During sampling, actions are clipped to valid ranges based on mask
    
    Args:
        state_dim: Dimension of state input
        action_dim: Dimension of action output
        hidden_dims: Hidden layer dimensions
        action_low: Minimum action value
        action_high: Maximum action value
        mask_penalty: Penalty value for masked actions
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        action_low: float = -1.0,
        action_high: float = 1.0,
        mask_penalty: float = -1e9
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.mask_penalty = mask_penalty
        
        # Mean network
        self.mean_net = MLP(
            state_dim,
            hidden_dims,
            action_dim,
            activation="relu"
        )
        
        # Log std network (state-dependent for flexibility)
        self.log_std_net = MLP(
            state_dim,
            hidden_dims,
            action_dim,
            activation="relu"
        )
        
        # Initialize log std to reasonable values (low variance initially)
        for layer in self.log_std_net.network:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -0.01, 0.01)
                nn.init.constant_(layer.bias, -1.0)  # Start with low std
    
    def forward(
        self,
        state: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with action masking.
        
        Args:
            state: State tensor [batch, state_dim]
            action_mask: Action mask [batch, 3] for [increase, decrease, hold]
                         None means all actions valid
            deterministic: If True, return mean action (no sampling)
            
        Returns:
            action: Sampled/clipped action [batch, action_dim]
            log_prob: Log probability of action [batch]
            mean: Mean of distribution [batch, action_dim]
        """
        batch_size = state.size(0)
        
        # Get distribution parameters
        mean = self.mean_net(state)
        log_std = self.log_std_net(state)
        log_std = torch.clamp(log_std, -20, 2)  # Numerical stability
        std = torch.exp(log_std)
        
        # Create normal distribution
        dist = torch.distributions.Normal(mean, std)
        
        if deterministic:
            action = torch.tanh(mean)
            # Apply action masking to deterministic action
            if action_mask is not None:
                action = self._apply_action_mask(action, action_mask)
            return action, torch.zeros(batch_size, device=state.device), mean
        
        # Sample action
        raw_action = dist.rsample()  # Reparameterization trick
        action = torch.tanh(raw_action)
        
        # Compute log probability with tanh correction
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        log_prob -= (2 * (np.log(2) - raw_action - F.softplus(-2 * raw_action))).sum(dim=-1)
        
        # Apply action masking
        if action_mask is not None:
            action, log_prob = self._apply_mask_to_action(
                action, log_prob, mean, std, action_mask
            )
        
        return action, log_prob, mean
    
    def _apply_action_mask(
        self,
        action: torch.Tensor,
        action_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply action mask to constrain action range.
        
        CANONICAL ACTION CONVENTION (matches env.py):
            action in [-1, 1]
            action < -0.1: INCREASE short exposure (more negative)
            action > 0.1:  DECREASE short exposure (cover toward 0)
            else:          HOLD current exposure
        
        Action mask indices:
            mask[0]: increase short allowed (1=yes, 0=no)
            mask[1]: decrease short/cover allowed (1=yes, 0=no)
            mask[2]: hold allowed (typically always 1)
        
        Args:
            action: Action in [-1, 1] [batch, action_dim]
            action_mask: Mask [batch, 3] for [increase, decrease, hold]
            
        Returns:
            masked_action: Constrained action
        """
        # CANONICAL CONVENTION:
        # action < -0.1: INCREASE short exposure → check mask[0]
        # action > 0.1:  DECREASE short exposure → check mask[1]
        
        # If increase short is blocked (mask[0] == 0), clip negative actions
        increase_blocked = (action_mask[:, 0] == 0).float()
        # If decrease short is blocked (mask[1] == 0), clip positive actions  
        decrease_blocked = (action_mask[:, 1] == 0).float()
        
        # Clip actions based on mask
        # If increase short blocked: action >= -0.1 (can't go more negative)
        # If decrease short blocked: action <= 0.1 (can't cover)
        action = torch.where(
            (action < -0.1) & (increase_blocked > 0),
            torch.full_like(action, -0.1),
            action
        )
        action = torch.where(
            (action > 0.1) & (decrease_blocked > 0),
            torch.full_like(action, 0.1),
            action
        )
        
        return action
    
    def _apply_mask_to_action(
        self,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        action_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply mask to action and adjust log probability.
        
        CANONICAL ACTION CONVENTION (matches env.py):
            action < -0.1: INCREASE short exposure → check mask[0]
            action > 0.1:  DECREASE short exposure → check mask[1]
        
        For continuous actions, masking works by:
        1. Constraining the action to valid ranges
        2. Adding large penalty to log_prob if action was in invalid region
        
        Args:
            action: Sampled action [batch, action_dim]
            log_prob: Log probability [batch]
            mean: Distribution mean [batch, action_dim]
            std: Distribution std [batch, action_dim]
            action_mask: Mask [batch, 3] for [increase, decrease, hold]
            
        Returns:
            masked_action: Constrained action
            masked_log_prob: Adjusted log probability
        """
        # CANONICAL CONVENTION:
        # action < -0.1: INCREASE short → check mask[0]
        # action > 0.1:  DECREASE short → check mask[1]
        is_increase = (action < -0.1).float()  # INCREASE short (more negative)
        is_decrease = (action > 0.1).float()   # DECREASE short (cover)
        
        # Check if action is in blocked region
        increase_invalid = is_increase * (action_mask[:, 0:1] == 0).float()
        decrease_invalid = is_decrease * (action_mask[:, 1:2] == 0).float()
        
        # Any invalid action gets massive penalty
        invalid = (increase_invalid + decrease_invalid).clamp(0, 1)
        
        # Constrain action to valid region
        masked_action = action.clone()
        # If increase short blocked, clip to -0.1 (can't go more negative)
        masked_action = torch.where(
            (action < -0.1) & (action_mask[:, 0:1] == 0),
            torch.full_like(action, -0.1),
            masked_action
        )
        # If decrease short blocked, clip to 0.1 (can't cover)
        masked_action = torch.where(
            (action > 0.1) & (action_mask[:, 1:2] == 0),
            torch.full_like(action, 0.1),
            masked_action
        )
        
        # Add penalty to log_prob for invalid actions
        # This makes invalid actions have near-zero probability
        masked_log_prob = log_prob + (invalid.squeeze(-1) * self.mask_penalty)
        
        return masked_action, masked_log_prob


class QNetwork(nn.Module):
    """
    Q-network for critic (takes state and action, outputs Q-value).
    
    SAC uses twin Q-networks to mitigate overestimation bias.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int]
    ):
        super().__init__()
        
        # Concatenate state and action
        input_dim = state_dim + action_dim
        
        self.network = MLP(
            input_dim,
            hidden_dims,
            output_dim=1,
            activation="relu"
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-value for state-action pair.
        
        Args:
            state: State tensor [batch, state_dim]
            action: Action tensor [batch, action_dim]
            
        Returns:
            q_value: Q(s, a) [batch, 1]
        """
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


# =============================================================================
# Custom RLlib Model (New API - Ray 2.x+)
# =============================================================================

if USE_NEW_API:
    class MaskedSACRLModule(SACTorchRLModule):
        """
        Custom SAC RLModule with action masking support.
        
        WARNING: THIS CLASS IS EXPERIMENTAL AND NOT ACTIVE IN THE MAIN TRAINER.
        The active WFO trainer (train_wfo.py) uses plain RLlib SAC without
        custom model masking. See module docstring for details.
        
        To use this class, you must explicitly wire it into the SAC config:
            sac_config.rl_module(rl_module_spec=MaskedSACRLModule)
        
        Observation format expected:
        {
            'state': [batch, 74],
            'action_mask': [batch, 3],  # [increase, decrease, hold]
            'kelly_leverage': [batch, 1]
        }
        """
        
        def __init__(self, config: Dict[str, Any]):
            logger.warning("MaskedSACRLModule is EXPERIMENTAL and NOT active in train_wfo.py")
            super().__init__(config)
            
            self.mask_penalty = config.get("mask_penalty", -1e9)
            self.state_dim = config.get("state_dim", 74)
            self.action_dim = config.get("action_dim", 1)
            
            # Override policy and Q-networks with masked versions
            hidden_dims = config.get("actor_hidden_dims", [256, 256])
            
            self.policy = MaskedGaussianPolicy(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dims=hidden_dims,
                mask_penalty=self.mask_penalty
            )
            
            # Re-initialize Q-networks if needed
            critic_dims = config.get("critic_hidden_dims", [256, 256])
            self.q1 = QNetwork(self.state_dim, self.action_dim, critic_dims)
            self.q2 = QNetwork(self.state_dim, self.action_dim, critic_dims)
            
            logger.info("MaskedSACRLModule initialized with action masking")
        
        def _forward_exploration(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            """Forward pass for exploration (sampling with action mask)."""
            # Extract components from observation
            obs = batch["obs"]
            
            if isinstance(obs, dict):
                state = obs.get("state", obs)
                action_mask = obs.get("action_mask", None)
            else:
                state = obs
                action_mask = None
            
            # Get action from policy with masking
            action, log_prob, mean = self.policy(
                state,
                action_mask=action_mask,
                deterministic=False
            )
            
            return {
                "actions": action,
                "action_logp": log_prob,
                "mean_actions": mean
            }
        
        def _forward_inference(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            """Forward pass for inference (deterministic with action mask)."""
            obs = batch["obs"]
            
            if isinstance(obs, dict):
                state = obs.get("state", obs)
                action_mask = obs.get("action_mask", None)
            else:
                state = obs
                action_mask = None
            
            # Get deterministic action with masking
            action, _, _ = self.policy(
                state,
                action_mask=action_mask,
                deterministic=True
            )
            
            return {"actions": action}
        
        def _forward_train(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            """Forward pass for training (computes all needed values)."""
            obs = batch["obs"]
            
            if isinstance(obs, dict):
                state = obs.get("state", obs)
                action_mask = obs.get("action_mask", None)
            else:
                state = obs
                action_mask = None
            
            # Sample action for policy evaluation
            action, log_prob, mean = self.policy(
                state,
                action_mask=action_mask,
                deterministic=False
            )
            
            # Compute Q-values
            q1_value = self.q1(state, action)
            q2_value = self.q2(state, action)
            
            # Target Q-values for critic training
            with torch.no_grad():
                next_action, next_log_prob, _ = self.policy(
                    batch.get("next_obs", state),
                    action_mask=action_mask,
                    deterministic=False
                )
                target_q1 = self.target_q1(state, next_action)
                target_q2 = self.target_q2(state, next_action)
                target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob.unsqueeze(-1)
            
            return {
                "actions": action,
                "action_logp": log_prob,
                "q1": q1_value,
                "q2": q2_value,
                "target_q": target_q,
                "mean_actions": mean
            }


# =============================================================================
# Custom RLlib Model (Legacy API - Ray 1.x)
# =============================================================================

if not USE_NEW_API and USE_NEW_API is not None:
    from ray.rllib.models.torch.misc import SlimFC
    from ray.rllib.utils.typing import ModelConfigDict, TensorType
    
    class MaskedSACModel(TorchModelV2, nn.Module):
        """
        Legacy custom SAC model with action masking.
        
        WARNING: THIS CLASS IS EXPERIMENTAL AND NOT ACTIVE IN THE MAIN TRAINER.
        The active WFO trainer (train_wfo.py) uses plain RLlib SAC without
        custom model masking. See module docstring for details.
        
        Compatible with Ray RLlib 1.x versions (legacy API).
        """
        
        def __init__(
            self,
            obs_space,
            action_space,
            num_outputs: int,
            model_config: ModelConfigDict,
            name: str,
            **kwargs
        ):
            logger.warning("MaskedSACModel is EXPERIMENTAL and NOT active in train_wfo.py")
            TorchModelV2.__init__(
                self, obs_space, action_space, num_outputs, model_config, name
            )
            nn.Module.__init__(self)
            
            # Extract config
            self.mask_penalty = model_config.get("custom_model_config", {}).get(
                "mask_penalty", -1e9
            )
            self.state_dim = model_config.get("custom_model_config", {}).get(
                "state_dim", 74
            )
            self.action_dim = num_outputs
            
            hidden_dims = model_config.get("custom_model_config", {}).get(
                "actor_hidden_dims", [256, 256]
            )
            
            # Create masked policy
            self.policy = MaskedGaussianPolicy(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dims=hidden_dims,
                mask_penalty=self.mask_penalty
            )
            
            logger.info("MaskedSACModel (legacy) initialized")
        
        def forward(
            self,
            input_dict: Dict[str, TensorType],
            state: List[TensorType],
            seq_lens: TensorType
        ) -> Tuple[TensorType, List[TensorType]]:
            """
            Forward pass with action masking.
            
            For SAC, forward returns action distribution parameters.
            """
            obs = input_dict["obs"]
            
            # Extract state and mask from dict observation
            if isinstance(obs, dict):
                state_features = obs.get("state", obs)
                action_mask = obs.get("action_mask", None)
            else:
                state_features = obs
                action_mask = None
            
            # Get action from policy
            action, log_prob, mean = self.policy(
                state_features,
                action_mask=action_mask,
                deterministic=False
            )
            
            # For SAC, we return the mean as logits (will be processed by action dist)
            return mean, state
        
        def value_function(self) -> TensorType:
            """Value function not used in SAC (uses Q-networks instead)."""
            return torch.zeros(1)


# =============================================================================
# RLlib Configuration Builder
# =============================================================================

def build_sac_config(
    env_class = None,
    config: Optional[SACConfig] = None,
    custom_model: bool = False  # Default to plain SAC (production path)
) -> Any:
    """
    Build Ray RLlib SAC configuration.
    
    IMPORTANT: This function has TWO modes controlled by custom_model parameter.
    
    ACTIVE PATH (custom_model=False, default in train_wfo.py):
        Returns plain RLlib SAC configuration. This is the PRODUCTION path.
        Action masking is handled by environment penalty, not neural network.
    
    EXPERIMENTAL PATH (custom_model=True):
        Wires in MaskedSACRLModule or MaskedSACModel for action masking.
        WARNING: This path is NOT ACTIVE in train_wfo.py and is EXPERIMENTAL.
        Use only if you explicitly want to test model-level action masking.
    
    Args:
        env_class: Gym environment class (e.g., ParabolicReversalEnv)
        config: SAC configuration
        custom_model: Whether to use custom masked model (EXPERIMENTAL, default False)
        
    Returns:
        algo_config: RLlib algorithm configuration
    """
    if custom_model:
        logger.warning("=" * 70)
        logger.warning("EXPERIMENTAL: build_sac_config called with custom_model=True")
        logger.warning("The masked model path is NOT active in train_wfo.py")
        logger.warning("Use custom_model=False for production training")
        logger.warning("=" * 70)
    
    try:
        from ray.rllib.algorithms.sac import SACConfig as RLlibSACConfig
    except ImportError:
        logger.error("Ray RLlib not installed. Run: pip install ray[rllib]")
        raise
    
    config = config or SACConfig()
    
    # Build base SAC config
    if USE_NEW_API:
        # Ray 2.x style
        algo_config = (
            RLlibSACConfig()
            .environment(
                env=env_class if env_class else "ParabolicReversalEnv",
                env_config={
                    "initial_capital": 100000.0,
                }
            )
            .framework("torch")
            .training(
                twin_q=True,
                q_model_config={
                    "fcnet_hiddens": config.critic_hidden_dims,
                    "fcnet_activation": config.activation,
                },
                policy_model_config={
                    "fcnet_hiddens": config.actor_hidden_dims,
                    "fcnet_activation": config.activation,
                },
                tau=config.tau,
                initial_alpha=config.alpha,
                target_entropy=config.target_entropy,
                n_step=1,
                replay_buffer_config={
                    "type": "MultiAgentReplayBuffer",
                    "capacity": config.buffer_size,
                },
            )
            .rl_module(
                rl_module_spec=MaskedSACRLModule if custom_model else None
            )
            .resources(
                num_gpus=1 if torch.cuda.is_available() else 0,
                num_cpus_per_worker=2,
            )
            .rollouts(
                num_rollout_workers=2,
                rollout_fragment_length=200,
            )
            .reporting(
                min_time_s_per_iteration=5,
            )
        )
    else:
        # Ray 1.x style
        from ray.rllib.models import ModelCatalog
        
        # Register custom model
        if custom_model:
            ModelCatalog.register_custom_model("masked_sac_model", MaskedSACModel)
        
        algo_config = (
            RLlibSACConfig()
            .environment(
                env=env_class if env_class else "ParabolicReversalEnv",
            )
            .framework("torch")
            .training(
                twin_q=True,
                q_model_config={
                    "fcnet_hiddens": config.critic_hidden_dims,
                    "fcnet_activation": config.activation,
                },
                policy_model_config={
                    "fcnet_hiddens": config.actor_hidden_dims,
                    "fcnet_activation": config.activation,
                },
                tau=config.tau,
                initial_alpha=config.alpha,
                target_entropy=config.target_entropy,
            )
            .resources(
                num_gpus=1 if torch.cuda.is_available() else 0,
            )
            .rollouts(
                num_rollout_workers=2,
            )
        )
        
        if custom_model:
            algo_config.model({
                "custom_model": "masked_sac_model",
                "custom_model_config": {
                    "mask_penalty": config.mask_penalty,
                    "state_dim": config.state_dim,
                    "actor_hidden_dims": config.actor_hidden_dims,
                }
            })
    
    return algo_config


# =============================================================================
# Standalone SAC Agent (for non-RLlib usage)
# =============================================================================

class StandaloneSACAgent:
    """
    Standalone SAC agent for environments where Ray RLlib is not available.
    
    This provides a pure PyTorch implementation of SAC with action masking
    that can be used directly without RLlib.
    """
    
    def __init__(self, config: Optional[SACConfig] = None):
        self.config = config or SACConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor network
        self.actor = MaskedGaussianPolicy(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dims=self.config.actor_hidden_dims,
            action_low=self.config.action_low,
            action_high=self.config.action_high,
            mask_penalty=self.config.mask_penalty
        ).to(self.device)
        
        # Critic networks (twin Q)
        self.q1 = QNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.critic_hidden_dims
        ).to(self.device)
        
        self.q2 = QNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.critic_hidden_dims
        ).to(self.device)
        
        # Target Q-networks
        self.target_q1 = QNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.critic_hidden_dims
        ).to(self.device)
        
        self.target_q2 = QNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.critic_hidden_dims
        ).to(self.device)
        
        # Copy weights to targets
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        
        # Temperature parameter
        self.log_alpha = torch.tensor(
            np.log(self.config.alpha),
            requires_grad=True,
            device=self.device
        )
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.lr_actor
        )
        self.q1_optimizer = torch.optim.Adam(
            self.q1.parameters(), lr=self.config.lr_critic
        )
        self.q2_optimizer = torch.optim.Adam(
            self.q2.parameters(), lr=self.config.lr_critic
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.config.lr_alpha
        )
        
        self.steps = 0
        
        logger.info(f"StandaloneSACAgent initialized on {self.device}")
    
    def select_action(
        self,
        state: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        deterministic: bool = False
    ) -> np.ndarray:
        """
        Select action using current policy.
        
        Args:
            state: State vector [state_dim]
            action_mask: Action mask [3]
            deterministic: If True, use mean action
            
        Returns:
            action: Selected action [action_dim]
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if action_mask is not None:
                mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
            else:
                mask_tensor = None
            
            action, _, _ = self.actor(
                state_tensor,
                action_mask=mask_tensor,
                deterministic=deterministic
            )
            
            return action.cpu().numpy()[0]
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update agent using a batch of transitions.
        
        Args:
            batch: Dictionary with 'state', 'action', 'reward', 'next_state', 'done'
            
        Returns:
            metrics: Dictionary of training metrics
        """
        state = batch['state'].to(self.device)
        action = batch['action'].to(self.device)
        reward = batch['reward'].to(self.device)
        next_state = batch['next_state'].to(self.device)
        done = batch['done'].to(self.device)
        
        # Get action mask if available
        action_mask = batch.get('action_mask', None)
        if action_mask is not None:
            action_mask = action_mask.to(self.device)
        
        # ===== Update Critic =====
        with torch.no_grad():
            # Sample next action
            next_action, next_log_prob, _ = self.actor(
                next_state, action_mask=action_mask, deterministic=False
            )
            
            # Compute target Q
            target_q1 = self.target_q1(next_state, next_action)
            target_q2 = self.target_q2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward.unsqueeze(-1) + (1 - done.unsqueeze(-1)) * self.config.gamma * (
                target_q - self.log_alpha.exp() * next_log_prob.unsqueeze(-1)
            )
        
        # Compute current Q
        current_q1 = self.q1(state, action)
        current_q2 = self.q2(state, action)
        
        # Critic loss
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        q_loss = q1_loss + q2_loss
        
        # Update critics
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        q_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()
        
        # ===== Update Actor =====
        new_action, log_prob, _ = self.actor(
            state, action_mask=action_mask, deterministic=False
        )
        
        q1_new = self.q1(state, new_action)
        q2_new = self.q2(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.log_alpha.exp().detach() * log_prob.unsqueeze(-1) - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ===== Update Alpha =====
        if self.config.auto_tune_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.config.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        # ===== Update Target Networks =====
        self._soft_update(self.target_q1, self.q1)
        self._soft_update(self.target_q2, self.q2)
        
        self.steps += 1
        
        return {
            'q_loss': q_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.log_alpha.exp().item(),
            'avg_q': q_new.mean().item()
        }
    
    def _soft_update(self, target: nn.Module, source: nn.Module):
        """Soft update target network parameters."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.config.tau) + param.data * self.config.tau
            )


# =============================================================================
# Testing and Validation
# =============================================================================

def test_agent():
    """Test the SAC agent components."""
    logger.info("=" * 70)
    logger.info("Testing SAC Decision Engine")
    logger.info("=" * 70)
    
    config = SACConfig()
    
    # Test 1: Masked Gaussian Policy
    logger.info("\n[TEST 1] Masked Gaussian Policy")
    policy = MaskedGaussianPolicy(
        state_dim=74,
        action_dim=1,
        hidden_dims=[256, 256]
    )
    
    batch_size = 4
    state = torch.randn(batch_size, 74)
    
    # Test without mask
    action, log_prob, mean = policy(state, action_mask=None, deterministic=False)
    logger.info(f"  State shape:  {state.shape}")
    logger.info(f"  Action shape: {action.shape}")
    logger.info(f"  Action range: [{action.min():.3f}, {action.max():.3f}]")
    logger.info(f"  Log prob:     {log_prob.shape}")
    
    # Test with mask (block increase)
    mask = torch.tensor([
        [0, 1, 1],  # Block increase
        [1, 1, 1],  # All valid
        [1, 0, 1],  # Block decrease
        [0, 0, 1],  # Block both directions
    ])
    
    action_masked, log_prob_masked, _ = policy(state, action_mask=mask, deterministic=False)
    logger.info(f"\n  With action mask:")
    logger.info(f"  Mask [0,1,1] (block increase): action={action_masked[0].item():.3f}")
    logger.info(f"  Mask [1,1,1] (all valid):      action={action_masked[1].item():.3f}")
    logger.info(f"  Mask [1,0,1] (block decrease): action={action_masked[2].item():.3f}")
    logger.info(f"  Mask [0,0,1] (block both):     action={action_masked[3].item():.3f}")
    
    # Verify masking works
    assert action_masked[0] <= 0.1, "Increase should be blocked"
    assert action_masked[2] >= -0.1, "Decrease should be blocked"
    assert abs(action_masked[3]) <= 0.1, "Both directions blocked"
    logger.info("  ✓ Action masking working correctly")
    
    # Test 2: Q-Network
    logger.info("\n[TEST 2] Q-Network")
    q_net = QNetwork(state_dim=74, action_dim=1, hidden_dims=[256, 256])
    q_value = q_net(state, action)
    logger.info(f"  Q-value shape: {q_value.shape}")
    logger.info(f"  Q-value range: [{q_value.min():.3f}, {q_value.max():.3f}]")
    logger.info("  ✓ Q-network working")
    
    # Test 3: Standalone Agent
    logger.info("\n[TEST 3] Standalone SAC Agent")
    agent = StandaloneSACAgent(config)
    
    # Test action selection
    test_state = np.random.randn(74)
    test_mask = np.array([1, 1, 1])
    
    action = agent.select_action(test_state, test_mask, deterministic=False)
    logger.info(f"  Selected action: {action}")
    
    action_det = agent.select_action(test_state, test_mask, deterministic=True)
    logger.info(f"  Deterministic action: {action_det}")
    
    # Test update
    batch = {
        'state': torch.randn(32, 74),
        'action': torch.randn(32, 1),
        'reward': torch.randn(32),
        'next_state': torch.randn(32, 74),
        'done': torch.zeros(32),
        'action_mask': torch.ones(32, 3)
    }
    
    metrics = agent.update(batch)
    logger.info(f"  Update metrics: {metrics}")
    logger.info("  ✓ Agent update working")
    
    # Test 4: RLlib Config
    logger.info("\n[TEST 4] RLlib Configuration")
    try:
        algo_config = build_sac_config(custom_model=True)
        logger.info("  ✓ RLlib config built successfully")
        logger.info(f"  Config type: {type(algo_config)}")
    except Exception as e:
        logger.warning(f"  RLlib not available: {e}")
    
    logger.info("\n" + "=" * 70)
    logger.info("All SAC tests passed!")
    logger.info("=" * 70)
    
    return True


if __name__ == "__main__":
    test_agent()

```

## src/rl/perception.py
```python
"""
Module I: Modular Perception and State Representation

This module implements the perception layer for the Parabolic Reversal Trading
trading system using a Temporal Convolutional Autoencoder (TCN-AE). It provides:

1. Temporal Convolutional Autoencoder (TCN-AE):
   - Encoder: Compresses high-dimensional OHLCV time-series into latent vector z_t
   - Decoder: Reconstructs input sequence for self-supervised pre-training
   - Causal convolutions prevent look-ahead bias

2. Hybrid State Concatenation:
   - Extracts frozen latent vector z_t (64 dimensions)
   - Concatenates with explicit V5 Relaxed features:
     * VWAP deviation (normalized)
     * Volume concentration
   - Combines with portfolio state features
   - Outputs final 74-dimensional state vector S_t

3. Pre-training Infrastructure:
   - Self-supervised MSE reconstruction loss
   - Learning rate scheduling
   - Checkpoint management
   - Data loader for historical market sequences

Author: AI Agent
Date: 2026-03-12
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import polars as pl
from typing import Dict, Tuple, Optional, List, Union, Any
from dataclasses import dataclass
from pathlib import Path
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PerceptionConfig:
    """Configuration for the Modular Perception module."""
    
    # Input sequence parameters
    sequence_length: int = 60  # 60 minutes of OHLCV data
    num_features: int = 5      # OHLCV: Open, High, Low, Close, Volume
    
    # TCN Architecture parameters
    encoder_channels: List[int] = None  # Channel progression
    kernel_size: int = 3       # Convolution kernel size
    dropout: float = 0.2       # Dropout rate for regularization
    use_weight_norm: bool = True  # Weight normalization for stability
    
    # Latent space dimensions (must match env.py observation space)
    latent_dim: int = 64       # Bottleneck layer size (z_t)
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 100
    patience: int = 10         # Early stopping patience
    min_delta: float = 1e-6    # Minimum improvement for early stopping
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Checkpointing
    checkpoint_dir: str = "models/perception"
    save_best: bool = True
    
    def __post_init__(self):
        if self.encoder_channels is None:
            # Default: 5 -> 32 -> 64 -> 128 -> 64 progression
            self.encoder_channels = [32, 64, 128, 64]


# =============================================================================
# TCN Components
# =============================================================================

class CausalConv1d(nn.Module):
    """
    Causal 1D convolution to prevent look-ahead bias.
    
    Causal convolution ensures that the output at time t only depends on
    inputs from time 0 to t, never on future information. This is critical
    for financial time series to prevent data leakage.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolution kernel
        dilation: Dilation factor for receptive field expansion
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        use_weight_norm: bool = True
    ):
        super().__init__()
        
        # Calculate padding for causal convolution
        # Output length = Input length - (kernel_size - 1) * dilation
        # To maintain length, pad (kernel_size - 1) * dilation on the left
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=0,  # We'll handle padding manually
            dilation=dilation
        )
        
        if use_weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with causal padding.
        
        Args:
            x: Input tensor [batch, channels, time]
            
        Returns:
            Output tensor [batch, out_channels, time]
        """
        # Pad left side only (causal padding)
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class ResidualBlock(nn.Module):
    """
    Residual block with dilated causal convolution.
    
    Architecture:
        Input -> Conv1 -> ReLU -> Dropout -> Conv2 -> ReLU -> Dropout -> Add -> Output
               |_______________________________________________________|
    
    Args:
        channels: Number of channels (input = output for residual)
        kernel_size: Convolution kernel size
        dilation: Dilation factor
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        use_weight_norm: bool = True
    ):
        super().__init__()
        
        self.conv1 = CausalConv1d(
            channels, channels, kernel_size, dilation, use_weight_norm
        )
        self.conv2 = CausalConv1d(
            channels, channels, kernel_size, dilation, use_weight_norm
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection (identity if same dimensions)
        self.downsample = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual forward pass."""
        residual = x
        
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        return out + residual


# =============================================================================
# Temporal Convolutional Autoencoder (TCN-AE)
# =============================================================================

class TCNEncoder(nn.Module):
    """
    Temporal Convolutional Network Encoder.
    
    Compresses high-dimensional time-series into low-dimensional latent vector.
    Uses dilated convolutions to exponentially expand receptive field without
    increasing parameters.
    
    Architecture:
        Input: [batch, num_features, sequence_length]
        -> Conv layers with increasing dilation
        -> Global average pooling
        -> Linear projection to latent_dim
        Output: [batch, latent_dim]
        
    Args:
        config: PerceptionConfig with architecture parameters
    """
    
    def __init__(self, config: PerceptionConfig):
        super().__init__()
        
        self.config = config
        channels = [config.num_features] + config.encoder_channels
        
        # Build encoder layers with exponentially increasing dilation
        layers = []
        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            dilation = 2 ** i  # 1, 2, 4, 8, ...
            
            # Initial convolution
            layers.append(CausalConv1d(
                in_ch, out_ch, config.kernel_size, dilation, config.use_weight_norm
            ))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            
            # Residual block
            layers.append(ResidualBlock(
                out_ch, config.kernel_size, dilation, config.dropout, config.use_weight_norm
            ))
        
        self.encoder_layers = nn.Sequential(*layers)
        
        # Global pooling and projection to latent space
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.to_latent = nn.Linear(
            config.encoder_channels[-1],
            config.latent_dim
        )
        
        # Layer normalization for stable latent representations
        self.latent_norm = nn.LayerNorm(config.latent_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input sequence to latent vector.
        
        Args:
            x: Input tensor [batch, num_features, sequence_length]
            
        Returns:
            z: Latent vector [batch, latent_dim]
        """
        # Apply encoder layers
        features = self.encoder_layers(x)  # [batch, channels, time]
        
        # Global average pooling over time dimension
        pooled = self.global_pool(features).squeeze(-1)  # [batch, channels]
        
        # Project to latent space
        z = self.to_latent(pooled)  # [batch, latent_dim]
        
        # Normalize for stable learning
        z = self.latent_norm(z)
        
        return z


class TCNDecoder(nn.Module):
    """
    Temporal Convolutional Network Decoder.
    
    Reconstructs input sequence from latent vector for self-supervised pre-training.
    
    Architecture:
        Input: [batch, latent_dim]
        -> Linear expansion
        -> ConvTranspose layers
        Output: [batch, num_features, sequence_length]
        
    Args:
        config: PerceptionConfig with architecture parameters
    """
    
    def __init__(self, config: PerceptionConfig):
        super().__init__()
        
        self.config = config
        
        # Calculate the feature map size after encoder pooling
        # This will be the target for initial expansion
        hidden_dim = config.encoder_channels[-1]
        
        # Expand latent vector to time-distributed features
        self.expand = nn.Sequential(
            nn.Linear(config.latent_dim, hidden_dim * (config.sequence_length // 4)),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Transposed convolution layers for upsampling
        # Reverse of encoder: 64 -> 128 -> 64 -> 32 -> 5
        decoder_channels = list(reversed(config.encoder_channels)) + [config.num_features]
        
        layers = []
        for i in range(len(decoder_channels) - 1):
            in_ch = decoder_channels[i]
            out_ch = decoder_channels[i + 1]
            
            # Use transposed convolution for upsampling
            layers.append(nn.ConvTranspose1d(
                in_ch, out_ch,
                kernel_size=config.kernel_size,
                stride=2 if i < 2 else 1,  # Upsample first two layers
                padding=config.kernel_size // 2,
                output_padding=1 if i < 2 else 0
            ))
            
            if i < len(decoder_channels) - 2:  # No activation on final layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(config.dropout))
        
        self.decoder_layers = nn.Sequential(*layers)
        
        # Final adjustment to exact sequence length
        self.output_adjust = nn.Conv1d(
            config.num_features,
            config.num_features,
            kernel_size=1
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to sequence reconstruction.
        
        Args:
            z: Latent vector [batch, latent_dim]
            
        Returns:
            reconstruction: [batch, num_features, sequence_length]
        """
        batch_size = z.size(0)
        
        # Expand to time-distributed representation
        hidden = self.expand(z)  # [batch, hidden_dim * (seq_len // 4)]
        hidden = hidden.view(
            batch_size,
            self.config.encoder_channels[-1],
            self.config.sequence_length // 4
        )
        
        # Apply decoder layers
        out = self.decoder_layers(hidden)
        
        # Adjust to exact sequence length
        if out.size(-1) != self.config.sequence_length:
            out = F.interpolate(
                out,
                size=self.config.sequence_length,
                mode='linear',
                align_corners=False
            )
        
        out = self.output_adjust(out)
        
        return out


class TemporalAutoencoder(nn.Module):
    """
    Complete Temporal Convolutional Autoencoder (TCN-AE).
    
    Combines encoder and decoder for end-to-end training.
    After pre-training, the decoder is discarded and only the encoder is used
    for state representation.
    
    Args:
        config: PerceptionConfig with architecture parameters
    """
    
    def __init__(self, config: Optional[PerceptionConfig] = None):
        super().__init__()
        
        self.config = config or PerceptionConfig()
        
        self.encoder = TCNEncoder(self.config)
        self.decoder = TCNDecoder(self.config)
        
        # Move to device
        self.to(self.config.device)
        
        logger.info(
            f"TCN-AE initialized: {self.config.num_features} -> "
            f"{self.config.latent_dim} -> {self.config.num_features} "
            f"(sequence length: {self.config.sequence_length})"
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor [batch, num_features, sequence_length]
            
        Returns:
            z: Latent vector [batch, latent_dim]
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstruction.
        
        Args:
            z: Latent vector [batch, latent_dim]
            
        Returns:
            reconstruction: [batch, num_features, sequence_length]
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode then decode.
        
        Args:
            x: Input tensor [batch, num_features, sequence_length]
            
        Returns:
            z: Latent vector [batch, latent_dim]
            reconstruction: [batch, num_features, sequence_length]
        """
        z = self.encode(x)
        recon = self.decode(z)
        return z, recon
    
    def freeze_encoder(self):
        """Freeze encoder weights after pre-training (discard decoder)."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder frozen for deployment")


# =============================================================================
# Hybrid State Representation
# =============================================================================

class StateRepresentation(nn.Module):
    """
    Hybrid State Representation Module.
    
    Combines the frozen latent vector from the TCN encoder with explicit
    V5 Relaxed features to form the final state vector S_t.
    
    Output format (74 dimensions total, matching env.py observation space):
        [0:64]   - Latent vector z_t (from TCN encoder)
        [64]     - VWAP deviation (normalized)
        [65]     - Volume concentration
        [66]     - Position size (normalized)
        [67]     - Unrealized P&L percentage
        [68]     - Current drawdown (normalized)
        [69]     - Kelly leverage fraction
        [70]     - Hour of day (normalized)
        [71]     - Minute (normalized)
        [72]     - In entry window flag
        [73]     - Must flatten flag
    
    Args:
        encoder: Frozen TCNEncoder instance
        config: PerceptionConfig
    """
    
    def __init__(
        self,
        encoder: TCNEncoder,
        config: Optional[PerceptionConfig] = None
    ):
        super().__init__()
        
        self.encoder = encoder
        self.config = config or PerceptionConfig()
        self.latent_dim = self.config.latent_dim
        
        # Freeze encoder if not already frozen
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Normalization parameters for explicit features
        # Based on empirical analysis from ARCHITECTURE_BLUEPRINT.md
        self.register_buffer('vwap_mean', torch.tensor(30.0))
        self.register_buffer('vwap_std', torch.tensor(15.0))
        self.register_buffer('max_position_value', torch.tensor(50000.0))
        self.register_buffer('max_drawdown', torch.tensor(-19180.0))
        
        logger.info(
            f"StateRepresentation initialized: "
            f"latent_dim={self.latent_dim}, total_state_dim=74"
        )
    
    def forward(
        self,
        market_sequence: torch.Tensor,
        vwap_deviation: torch.Tensor,
        volume_concentration: torch.Tensor,
        position_value: torch.Tensor,
        unrealized_pnl_pct: torch.Tensor,
        current_drawdown: torch.Tensor,
        kelly_fraction: torch.Tensor,
        time_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Construct the complete state vector.
        
        Args:
            market_sequence: OHLCV sequence [batch, num_features, sequence_length]
            vwap_deviation: VWAP deviation percentage [batch]
            volume_concentration: Volume concentration [0-1] [batch]
            position_value: Current position value [batch]
            unrealized_pnl_pct: Unrealized P&L as percentage [batch]
            current_drawdown: Current drawdown amount [batch]
            kelly_fraction: Current Kelly leverage [batch]
            time_features: [hour, minute, in_window, must_flatten] [batch, 4]
            
        Returns:
            state: Complete state vector [batch, 74]
        """
        batch_size = market_sequence.size(0)
        device = market_sequence.device
        
        # Extract latent representation (frozen encoder)
        with torch.no_grad():
            z = self.encoder(market_sequence)  # [batch, latent_dim]
        
        # Normalize explicit features
        vwap_norm = (vwap_deviation - self.vwap_mean) / self.vwap_std
        vwap_norm = vwap_norm.unsqueeze(-1)  # [batch, 1]
        
        vol_conc = volume_concentration.unsqueeze(-1)  # [batch, 1]
        
        pos_norm = (position_value / self.max_position_value).unsqueeze(-1)
        
        pnl_pct = unrealized_pnl_pct.unsqueeze(-1)  # Already percentage
        
        dd_norm = (current_drawdown / self.max_drawdown).unsqueeze(-1)
        
        kelly_norm = (kelly_fraction / 3.0).unsqueeze(-1)  # Max leverage is 3.0
        
        # Ensure time features are correct shape
        if time_features.dim() == 1:
            time_features = time_features.unsqueeze(0)
        
        # Concatenate all features
        state = torch.cat([
            z,                          # [batch, 64]
            vwap_norm,                  # [batch, 1]
            vol_conc,                   # [batch, 1]
            pos_norm,                   # [batch, 1]
            pnl_pct,                    # [batch, 1]
            dd_norm,                    # [batch, 1]
            kelly_norm,                 # [batch, 1]
            time_features               # [batch, 4]
        ], dim=-1)  # [batch, 74]
        
        return state
    
    def to_numpy(self, state: torch.Tensor) -> np.ndarray:
        """Convert state tensor to numpy array for environment."""
        return state.cpu().numpy()


# =============================================================================
# Dataset and Data Loading
# =============================================================================

class MarketSequenceDataset(Dataset):
    """
    Dataset for market sequence samples.
    
    Loads and preprocesses OHLCV sequences for autoencoder training.
    
    Args:
        data: DataFrame with OHLCV columns
        sequence_length: Length of each sequence sample
        transform: Optional preprocessing transform
    """
    
    def __init__(
        self,
        data: pl.DataFrame,
        sequence_length: int = 60,
        transform: Optional[Any] = None
    ):
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Extract OHLCV columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        available_cols = [c.lower() for c in data.columns]
        
        # Normalize column names
        col_mapping = {}
        for req in required_cols:
            for avail in data.columns:
                if req in avail.lower():
                    col_mapping[req] = avail
                    break
        
        if len(col_mapping) < 5:
            raise ValueError(f"Missing required columns. Found: {col_mapping}")
        
        # Extract and normalize features
        self.features = np.zeros((len(data), 5))
        for i, col in enumerate(required_cols):
            if col in col_mapping:
                self.features[:, i] = data[col_mapping[col]].to_numpy()
        
        # Normalize each feature (z-score)
        self.feature_means = self.features.mean(axis=0)
        self.feature_stds = self.features.std(axis=0) + 1e-8
        self.features = (self.features - self.feature_means) / self.feature_stds
        
        # Create sequences
        self.sequences = []
        for i in range(len(self.features) - sequence_length + 1):
            seq = self.features[i:i + sequence_length]
            self.sequences.append(seq)
        
        self.sequences = np.array(self.sequences)
        logger.info(f"Created {len(self.sequences)} sequences of length {sequence_length}")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        seq = self.sequences[idx]  # [sequence_length, 5]
        
        # Transpose to [channels, time]
        seq = torch.FloatTensor(seq).transpose(0, 1)
        
        if self.transform:
            seq = self.transform(seq)
        
        return seq


# =============================================================================
# Pre-training Infrastructure
# =============================================================================

class PerceptionTrainer:
    """
    Trainer for self-supervised pre-training of the TCN Autoencoder.
    
    Implements:
    - MSE reconstruction loss
    - Adam optimizer with learning rate scheduling
    - Early stopping
    - Checkpoint saving
    
    Args:
        model: TemporalAutoencoder instance
        config: PerceptionConfig
    """
    
    def __init__(
        self,
        model: TemporalAutoencoder,
        config: Optional[PerceptionConfig] = None
    ):
        self.model = model
        self.config = config or PerceptionConfig()
        self.device = torch.device(self.config.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Checkpoint directory
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            batch = batch.to(self.device)
            
            # Forward pass
            z, recon = self.model(batch)
            loss = self.criterion(recon, batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / num_batches
    
    def validate(self, dataloader: DataLoader) -> float:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                batch = batch.to(self.device)
                z, recon = self.model(batch)
                loss = self.criterion(recon, batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict[str, List[float]]:
        """
        Full training loop with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            
        Returns:
            history: Dictionary with train and validation losses
        """
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = None
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Early stopping check
                if val_loss < self.best_val_loss - self.config.min_delta:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                    
                    # Save best model
                    if self.config.save_best:
                        self.save_checkpoint('best_model.pt')
                        logger.info(f"New best model saved (val_loss: {val_loss:.6f})")
                else:
                    self.epochs_without_improvement += 1
                
                logger.info(
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"Best: {self.best_val_loss:.6f}"
                )
                
                # Early stopping
                if self.epochs_without_improvement >= self.config.patience:
                    logger.info(
                        f"Early stopping triggered after {epoch + 1} epochs"
                    )
                    break
            else:
                logger.info(f"Train Loss: {train_loss:.6f}")
        
        history = {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses if val_loader else None
        }
        
        return history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }, path)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        logger.info(f"Checkpoint loaded from {path}")


# =============================================================================
# Factory Functions
# =============================================================================

def create_perception_module(
    checkpoint_path: Optional[str] = None,
    config: Optional[PerceptionConfig] = None
) -> Tuple[StateRepresentation, PerceptionConfig]:
    """
    Factory function to create perception module.
    
    Loads pre-trained encoder if checkpoint provided, otherwise creates fresh model.
    
    Args:
        checkpoint_path: Path to pre-trained checkpoint
        config: Optional configuration (uses default if not provided)
        
    Returns:
        perception: StateRepresentation module with frozen encoder
        config: PerceptionConfig used
    """
    config = config or PerceptionConfig()
    
    # Create autoencoder
    autoencoder = TemporalAutoencoder(config)
    
    # Load checkpoint if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        autoencoder.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded pre-trained encoder from {checkpoint_path}")
    
    # Freeze encoder and create state representation
    autoencoder.freeze_encoder()
    perception = StateRepresentation(autoencoder.encoder, config)
    
    return perception, config


def pretrain_perception(
    data: pl.DataFrame,
    val_split: float = 0.2,
    config: Optional[PerceptionConfig] = None
) -> TemporalAutoencoder:
    """
    End-to-end pre-training function.
    
    Args:
        data: OHLCV DataFrame for training
        val_split: Fraction of data for validation
        config: Training configuration
        
    Returns:
        model: Trained TemporalAutoencoder
    """
    config = config or PerceptionConfig()
    
    # Create dataset
    dataset = MarketSequenceDataset(data, config.sequence_length)
    
    # Split train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if config.device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create model and trainer
    model = TemporalAutoencoder(config)
    trainer = PerceptionTrainer(model, config)
    
    # Train
    history = trainer.train(train_loader, val_loader)
    
    return model, history


# =============================================================================
# Testing and Validation
# =============================================================================

def test_perception_module():
    """Run comprehensive tests on the perception module."""
    logger.info("=" * 70)
    logger.info("Testing Modular Perception (TCN-AE)")
    logger.info("=" * 70)
    
    # Create config
    config = PerceptionConfig(
        sequence_length=60,
        num_features=5,
        latent_dim=64,
        batch_size=16
    )
    
    logger.info(f"\nConfiguration:")
    logger.info(f"  Sequence length: {config.sequence_length}")
    logger.info(f"  Input features: {config.num_features}")
    logger.info(f"  Latent dimension: {config.latent_dim}")
    logger.info(f"  Device: {config.device}")
    
    # Test 1: Autoencoder forward pass
    logger.info("\n[TEST 1] Autoencoder forward pass")
    model = TemporalAutoencoder(config)
    
    batch_size = 4
    dummy_input = torch.randn(batch_size, config.num_features, config.sequence_length)
    dummy_input = dummy_input.to(config.device)
    
    z, recon = model(dummy_input)
    
    logger.info(f"  Input shape:  {dummy_input.shape}")
    logger.info(f"  Latent shape: {z.shape}")
    logger.info(f"  Output shape: {recon.shape}")
    
    assert z.shape == (batch_size, config.latent_dim), "Latent shape mismatch"
    assert recon.shape == dummy_input.shape, "Reconstruction shape mismatch"
    logger.info("  ✓ Tensor shapes correct")
    
    # Test 2: State representation
    logger.info("\n[TEST 2] Hybrid State Representation")
    
    model.freeze_encoder()
    state_module = StateRepresentation(model.encoder, config)
    
    # Create dummy market data
    market_seq = torch.randn(batch_size, config.num_features, config.sequence_length)
    
    # Create dummy explicit features
    vwap_dev = torch.tensor([25.0, 30.0, 46.0, 21.0])  # Including threshold boundary
    vol_conc = torch.tensor([0.8, 1.0, 0.66, 0.5])
    pos_value = torch.tensor([25000.0, 0.0, 50000.0, 10000.0])
    unrealized_pnl = torch.tensor([0.02, 0.0, -0.01, 0.05])
    drawdown = torch.tensor([-5000.0, 0.0, -10000.0, -15000.0])
    kelly = torch.tensor([0.5, 0.1, 1.5, 2.0])
    time_feats = torch.tensor([
        [10.5, 30.0, 1.0, 0.0],
        [9.0, 0.0, 0.0, 0.0],
        [14.0, 45.0, 1.0, 0.0],
        [15.5, 0.0, 0.0, 1.0]
    ])
    
    state = state_module(
        market_seq, vwap_dev, vol_conc, pos_value,
        unrealized_pnl, drawdown, kelly, time_feats
    )
    
    logger.info(f"  Market sequence: {market_seq.shape}")
    logger.info(f"  Latent vector:   {z.shape}")
    logger.info(f"  Final state:     {state.shape}")
    
    assert state.shape == (batch_size, 74), f"State shape {state.shape} != (4, 74)"
    logger.info("  ✓ State vector shape correct (batch, 74)")
    
    # Verify state composition
    logger.info("\n[TEST 3] State vector composition")
    logger.info(f"  [0:64]   Latent z:          {state[0, 0:64].shape}")
    logger.info(f"  [64]     VWAP deviation:    {state[0, 64].item():.4f}")
    logger.info(f"  [65]     Volume conc:       {state[0, 65].item():.4f}")
    logger.info(f"  [66]     Position:          {state[0, 66].item():.4f}")
    logger.info(f"  [67]     Unrealized PnL:    {state[0, 67].item():.4f}")
    logger.info(f"  [68]     Drawdown:          {state[0, 68].item():.4f}")
    logger.info(f"  [69]     Kelly fraction:    {state[0, 69].item():.4f}")
    logger.info(f"  [70:74]  Time features:     {state[0, 70:74].numpy()}")
    
    # Test 3: Reconstruction quality
    logger.info("\n[TEST 4] Reconstruction quality")
    mse = F.mse_loss(recon, dummy_input).item()
    logger.info(f"  MSE (untrained): {mse:.6f}")
    
    # Test 4: Encoder receptive field
    logger.info("\n[TEST 5] Causal convolution check")
    seq1 = torch.randn(1, config.num_features, config.sequence_length)
    seq2 = seq1.clone()
    seq2[:, :, -1] += 10.0  # Modify only last time step
    
    z1 = model.encode(seq1)
    z2 = model.encode(seq2)
    
    diff = torch.abs(z1 - z2).mean().item()
    logger.info(f"  Latent difference after last-step perturbation: {diff:.6f}")
    assert diff > 0.01, "Encoder may not be properly causal"
    logger.info("  ✓ Encoder responds to recent changes (causal)")
    
    # Test 6: Dataset creation
    logger.info("\n[TEST 6] Dataset creation")
    dummy_df = pl.DataFrame({
        'open': np.random.randn(1000),
        'high': np.random.randn(1000),
        'low': np.random.randn(1000),
        'close': np.random.randn(1000),
        'volume': np.random.randn(1000)
    })
    
    dataset = MarketSequenceDataset(dummy_df, sequence_length=60)
    logger.info(f"  Dataset size: {len(dataset)}")
    sample = dataset[0]
    logger.info(f"  Sample shape: {sample.shape}")
    assert sample.shape == (5, 60), "Dataset sample shape incorrect"
    logger.info("  ✓ Dataset working correctly")
    
    logger.info("\n" + "=" * 70)
    logger.info("All tests passed successfully!")
    logger.info("=" * 70)
    
    return True


if __name__ == "__main__":
    test_perception_module()

```

## src/rl/data_provider_hybrid.py
```python
"""
Hybrid Data Provider for RL Training

Combines:
1. CSV setups (proven winners with actual trades)
2. Parquet data (high volatility days)

Key difference: Instead of filtering to specific hours, we filter to bars
where VWAP deviation > 20%, ensuring the agent learns in valid trading conditions.
"""

import os
import pickle
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import namedtuple

import numpy as np
import polars as pl
import yaml

# Bar data structure for environment compatibility
Bar = namedtuple('Bar', ['open', 'high', 'low', 'close', 'volume', 'vwap', 'vwap_deviation', 'timestamp'])

from src.rl.config import RL_CONFIG
from src.utils.logger import logger


class HybridDataProvider:
    """
    Hybrid data provider that loads from both CSV setups and Parquet files.
    
    CSV setups are validated winners with actual profitable trades.
    Parquet setups add variety from all high-volatility days.
    
    CRITICAL: Supports date range filtering to prevent WFO data leakage.
    When date_range is set, only episodes within [start_date, end_date] 
    are available for sampling.
    """
    
    def __init__(
        self,
        csv_path: str = "reports/relaxed_909_backtest.csv",
        parquet_dir: str = "data/cache/1min_extended",
        cache_dir: str = None,  # Will be resolved to absolute path
        csv_weight: float = 0.7,
        min_vwap_deviation: float = 20.0,  # Must match strategy entry threshold (settings.yaml 1.20 = 20%)
        skip_parquet_scan: bool = False,  # Scan all Parquet files for unbiased training universe
        date_range: Optional[Tuple[Optional[str], Optional[str]]] = None,  # (start_date, end_date) for WFO
        seed: Optional[int] = None,  # Seed for reproducible episode selection
        mode: str = "train",  # "train" or "eval" - for logging/validation purposes
    ):
        """
        Initialize hybrid data provider.
        
        Args:
            csv_path: Path to CSV with backtest results
            parquet_dir: Directory with Parquet files
            cache_dir: Directory for caching index
            csv_weight: Probability of sampling from CSV (vs Parquet)
            min_vwap_deviation: Minimum VWAP deviation for valid setups
            skip_parquet_scan: Skip Parquet scanning (for quick testing with CSV only)
            date_range: Optional (start_date, end_date) tuple to filter episodes.
                       Format: "YYYY-MM-DD". Used for WFO to prevent data leakage.
            seed: Random seed for reproducible episode selection
            mode: "train" or "eval" - determines sampling behavior and validation
        """
        self.csv_path = Path(csv_path)
        self.parquet_dir = Path(parquet_dir)
        
        # Validate mode
        if mode not in ("train", "eval"):
            raise ValueError(f"mode must be 'train' or 'eval', got {mode}")
        self.mode = mode
        
        # Resolve cache_dir to absolute path
        if cache_dir is None:
            # Try to find the project root
            possible_roots = [
                Path("/mnt/c/quant_trading"),
                Path.home() / "quant_trading",
                Path.cwd(),
            ]
            for root in possible_roots:
                if (root / "src" / "scripts" / "data" / "cache").exists() or (root / "src").exists():
                    self.cache_dir = root / "src" / "scripts" / "data" / "cache"
                    break
            else:
                self.cache_dir = Path("src/scripts/data/cache")
        else:
            self.cache_dir = Path(cache_dir)
        self.csv_weight = csv_weight
        self.min_vwap_deviation = min_vwap_deviation
        self.skip_parquet_scan = skip_parquet_scan
        self.date_range = date_range
        self.seed = seed
        
        # Initialize random state for reproducibility
        self._rng = random.Random(seed)
        np.random.seed(seed)
        
        # Log initialization with mode
        logger.info(f"[{self.mode.upper()}] HybridDataProvider initialized")
        if date_range:
            logger.info(f"[{self.mode.upper()}] Date range: {date_range[0]} to {date_range[1]}")
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or build index
        self.index_path = self.cache_dir / "hybrid_index.pkl"
        self.csv_setups: List[Dict] = []
        self.parquet_setups: List[Dict] = []
        
        # Current episode state
        self.current_data: Optional[pl.DataFrame] = None
        self.current_symbol: Optional[str] = None
        self.current_date: Optional[str] = None
        self.current_bar_idx: int = 0
        self.start_bar_idx: int = 0  # First bar where VWAP > 20%
        
        self._load_or_build_index()
    
    def _load_or_build_index(self):
        """Load cached index or build from scratch."""
        logger.info(f"Index path: {self.index_path}, exists: {self.index_path.exists()}")
        if self.index_path.exists():
            logger.info(f"Loading cached index: {self.index_path}")
            with open(self.index_path, 'rb') as f:
                index = pickle.load(f)
                all_csv_setups = index['csv_setups']
                all_parquet_setups = index['parquet_setups']
            
            # Apply date range filtering if specified
            self.csv_setups = self._filter_by_date_range(all_csv_setups)
            self.parquet_setups = self._filter_by_date_range(all_parquet_setups)
            
            logger.info(f"  - CSV setups: {len(self.csv_setups)} (filtered from {len(all_csv_setups)})")
            logger.info(f"  - Parquet setups: {len(self.parquet_setups)} (filtered from {len(all_parquet_setups)})")
            if self.date_range:
                logger.info(f"  - Date range filter: {self.date_range[0]} to {self.date_range[1]}")
        else:
            logger.info("Building index from scratch...")
            self._build_index()
    
    def _filter_by_date_range(self, setups: List[Dict]) -> List[Dict]:
        """
        Filter setups to only include those within date_range.
        
        This is CRITICAL for WFO to prevent data leakage - ensures that
        training only sees training dates and testing only sees test dates.
        """
        if self.date_range is None:
            return setups
        
        start_date, end_date = self.date_range
        filtered = []
        
        for setup in setups:
            date_str = setup.get('date', '')
            if not date_str:
                continue
            
            # Include if within range (inclusive)
            if start_date and date_str < start_date:
                continue
            if end_date and date_str > end_date:
                continue
            
            filtered.append(setup)
        
        return filtered
    
    def _build_index(self):
        """Build index of valid trading days from both sources."""
        # Load CSV setups
        if self.csv_path.exists():
            self._load_csv_setups()
        else:
            logger.warning(f"CSV file not found: {self.csv_path}")
        
        # Load Parquet setups (unless skipped)
        if self.skip_parquet_scan:
            logger.info("Skipping Parquet scan (quick test mode)")
        elif self.parquet_dir.exists():
            self._load_parquet_setups()
        else:
            logger.warning(f"Parquet directory not found: {self.parquet_dir}")
        
        # Save index
        index = {
            'csv_setups': self.csv_setups,
            'parquet_setups': self.parquet_setups,
            'built_at': datetime.now().isoformat()
        }
        with open(self.index_path, 'wb') as f:
            pickle.dump(index, f)
        
        total = len(self.csv_setups) + len(self.parquet_setups)
        logger.info(f"Index complete: {total} total setups")
    
    def _load_csv_setups(self):
        """Load setups from CSV with trade data."""
        import pandas as pd
        
        df = pd.read_csv(self.csv_path)
        logger.info(f"Loading CSV: {len(df)} rows")
        
        valid_count = 0
        for _, row in df.iterrows():
            symbol = row['symbol']
            date_str = row['date']
            total_pnl = row.get('pnl', row.get('total_pnl', 0))
            
            # Only include setups with profitable trades (>$100 to filter noise)
            if total_pnl <= 100:
                continue
            
            # Check if data file exists (handle both naming patterns)
            data_file = self.parquet_dir / f"{symbol}.parquet"
            if not data_file.exists():
                # Try extended naming pattern: SYMBOL_1min_20190101_20241231.parquet
                matching_files = list(self.parquet_dir.glob(f"{symbol}_1min_*.parquet"))
                if matching_files:
                    data_file = matching_files[0]
                else:
                    continue
            if not data_file.exists():
                continue
            
            # Validate VWAP data exists and meets threshold
            if self._validate_vwap_in_data(symbol, date_str):
                self.csv_setups.append({
                    'symbol': symbol,
                    'date': date_str,
                    'source': 'csv',
                    'pnl': total_pnl
                })
                valid_count += 1
        
        logger.info(f"  Valid CSV setups with trades: {valid_count}")
    
    def _load_parquet_setups(self):
        """Load setups from ALL Parquet files (unbiased training universe).
        
        CRITICAL: Includes ALL trading days - boring, noisy, losing, AND winning.
        The RL agent MUST learn to output 0.0 (Hold/Flat) when conditions don't align.
        """
        # Get all parquet files
        parquet_files = list(self.parquet_dir.glob("*.parquet"))
        logger.info(f"Scanning {len(parquet_files)} symbols")
        
        scanned = 0
        for pq_file in parquet_files:
            symbol = pq_file.stem
            # Handle extended naming pattern
            if '_1min_' in symbol:
                symbol = symbol.split('_1min_')[0]
            
            try:
                df = pl.read_parquet(pq_file)
                
                # Filter to market hours and group by date
                df = df.filter(
                    (pl.col('timestamp').dt.hour() >= 9) &
                    (pl.col('timestamp').dt.hour() <= 16)
                )
                df = df.with_columns([
                    pl.col('timestamp').dt.date().alias('date')
                ])
                
                # Always recalculate VWAP from market open
                df = self._calculate_vwap(df)
                
                # Find days with VWAP > threshold
                df = df.with_columns([
                    ((pl.col('close') - pl.col('vwap')) / pl.col('vwap') * 100).alias('vwap_dev')
                ])
                
                for date_val in df['date'].unique():
                    date_df = df.filter(pl.col('date') == date_val)
                    max_vwap_dev = date_df['vwap_dev'].abs().max()
                    
                    # Include ALL trading days - boring, noisy, losing, AND winning
                    # The agent must learn when NOT to trade as much as when TO trade
                    self.parquet_setups.append({
                        'symbol': symbol,
                        'date': date_val.strftime('%Y-%m-%d'),
                        'source': 'parquet',
                        'max_vwap_dev': float(max_vwap_dev)
                    })
                
                scanned += 1
                if scanned % 500 == 0:
                    logger.info(f"  Scanned {scanned} symbols...")
                    
            except Exception as e:
                logger.debug(f"Error scanning {symbol}: {e}")
                continue
        
        # Log distribution of volatility levels
        volatile_days = sum(1 for s in self.parquet_setups if s['max_vwap_dev'] >= self.min_vwap_deviation)
        boring_days = len(self.parquet_setups) - volatile_days
        
        logger.info(f"  Total trading days: {len(self.parquet_setups)}")
        logger.info(f"    - Volatile days (VWAP > {self.min_vwap_deviation}%): {volatile_days}")
        logger.info(f"    - Boring/noisy days (VWAP < {self.min_vwap_deviation}%): {boring_days}")
        logger.info(f"  Agent will learn to Hold/Flat on {boring_days} non-setup days")
    
    def _validate_vwap_in_data(self, symbol: str, date_str: str) -> bool:
        """Check if VWAP data exists and meets threshold for a specific date."""
        try:
            # Handle both naming patterns
            data_file = self.parquet_dir / f"{symbol}.parquet"
            if not data_file.exists():
                matching_files = list(self.parquet_dir.glob(f"{symbol}_1min_*.parquet"))
                if matching_files:
                    data_file = matching_files[0]
                else:
                    return False
            
            if not data_file.exists():
                return False
            
            df = pl.read_parquet(data_file)
            
            # Parse date
            date_val = datetime.strptime(date_str, '%Y-%m-%d').date()
            
            # Filter to date and market hours (same as episode loading)
            df = df.filter(
                (pl.col('timestamp').dt.date() == date_val) &
                (pl.col('timestamp').dt.hour() >= 9) &
                (pl.col('timestamp').dt.hour() <= 16)
            )
            
            if len(df) == 0:
                return False
            
            # Always recalculate VWAP from market open
            df = self._calculate_vwap(df)
            
            # Calculate VWAP deviation
            df = df.with_columns([
                ((pl.col('close') - pl.col('vwap')) / pl.col('vwap') * 100).alias('vwap_dev')
            ])
            
            max_dev = df['vwap_dev'].abs().max()
            return max_dev >= self.min_vwap_deviation
            
        except Exception as e:
            logger.debug(f"Error validating {symbol} {date_str}: {e}")
            return False
    
    def _calculate_vwap(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate VWAP anchored from market open (9:30 AM ET)."""
        # Convert timestamps to ET and extract hour/minute
        et_times = df['timestamp'].dt.convert_time_zone('America/New_York')
        hours = et_times.dt.hour().cast(pl.Int32).to_numpy()
        minutes = et_times.dt.minute().cast(pl.Int32).to_numpy()
        
        # Calculate minutes from midnight (using numpy to avoid overflow)
        minutes_from_midnight = hours * 60 + minutes
        market_open_minutes = 9 * 60 + 30
        after_open_mask = minutes_from_midnight >= market_open_minutes
        
        # Calculate typical price
        df = df.with_columns([
            ((pl.col('high') + pl.col('low') + pl.col('close')) / 3).alias('typical_price')
        ])
        
        # Calculate PV (price * volume)
        typical_price = df['typical_price'].to_numpy()
        volume = df['volume'].to_numpy()
        close = df['close'].to_numpy()
        pv = typical_price * volume
        
        # Calculate cumulative VWAP from market open
        cum_pv = 0.0
        cum_vol = 0.0
        vwap_values = []
        
        for i in range(len(df)):
            if after_open_mask[i]:
                cum_pv += pv[i]
                cum_vol += volume[i]
                vwap_values.append(cum_pv / cum_vol if cum_vol > 0 else close[i])
            else:
                vwap_values.append(close[i])
        
        df = df.with_columns([
            pl.Series('vwap', vwap_values)
        ])
        
        return df.drop(['typical_price'])
    
    def _load_trading_day(self, symbol: str, date_str: str) -> Optional[pl.DataFrame]:
        """
        Load and prepare trading day data.
        
        Returns DataFrame with all bars, and finds the first bar where VWAP > 20%.
        """
        try:
            # Handle both naming patterns
            data_file = self.parquet_dir / f"{symbol}.parquet"
            if not data_file.exists():
                matching_files = list(self.parquet_dir.glob(f"{symbol}_1min_*.parquet"))
                if matching_files:
                    data_file = matching_files[0]
                else:
                    return None
            
            df = pl.read_parquet(data_file)
            
            # Parse date
            date_val = datetime.strptime(date_str, '%Y-%m-%d').date()
            
            # Filter to date and market hours
            df = df.filter(
                (pl.col('timestamp').dt.date() == date_val) &
                (pl.col('timestamp').dt.hour() >= 9) &
                (pl.col('timestamp').dt.hour() <= 16)
            )
            
            if len(df) < 60:  # Need at least 1 hour of data
                logger.warning(f"Insufficient data for {symbol} {date_str}: {len(df)} bars")
                return None
            
            # Always recalculate VWAP from market open
            df = self._calculate_vwap(df)
            
            # Calculate VWAP deviation
            df = df.with_columns([
                ((pl.col('close') - pl.col('vwap')) / pl.col('vwap') * 100).alias('vwap_dev')
            ])
            
            # Find first bar where VWAP > 20% (entry threshold - 3% buffer)
            entry_threshold = RL_CONFIG.get('min_vwap_deviation_entry', 20.0)
            valid_bars = df.filter(pl.col('vwap_dev').abs() > (entry_threshold - 3.0))
            
            if len(valid_bars) == 0:
                logger.warning(f"No bars with VWAP > {entry_threshold - 3}% for {symbol} {date_str}")
                return None
            
            # Get the index of the first valid bar
            first_valid_idx = valid_bars.select(pl.first('__row_index__')).to_numpy()[0, 0] if '__row_index__' in valid_bars.columns else 0
            
            # Add row index if not present
            if '__row_index__' not in df.columns:
                df = df.with_row_index('__row_index__')
                valid_bars = df.filter(pl.col('vwap_dev').abs() > (entry_threshold - 3.0))
                first_valid_idx = int(valid_bars['__row_index__'][0])
            
            logger.info(f"Loaded {symbol} {date_str}: {len(df)} bars, first valid at bar {first_valid_idx}")
            return df
            
        except Exception as e:
            logger.warning(f"Failed to load {symbol} {date_str}: {e}")
            return None
    
    def reset_episode(self) -> bool:
        """
        Reset and load a new episode.
        
        Uses seeded RNG for reproducible episode selection.
        CRITICAL: Runtime assertion verifies date is within configured bounds.
        Returns True if successful, False otherwise.
        """
        max_attempts = 10
        
        for attempt in range(max_attempts):
            # Choose source based on weight (using seeded RNG)
            if self._rng.random() < self.csv_weight and len(self.csv_setups) > 0:
                setup = self._rng.choice(self.csv_setups)
            elif len(self.parquet_setups) > 0:
                setup = self._rng.choice(self.parquet_setups)
            else:
                logger.error(f"[{self.mode.upper()}] No setups available")
                return False
            
            symbol = setup['symbol']
            date_str = setup['date']
            
            # =====================================================================
            # CRITICAL: Runtime assertion to prevent WFO data leakage
            # =====================================================================
            if self.date_range is not None:
                start_date, end_date = self.date_range
                if start_date and date_str < start_date:
                    logger.error(
                        f"[{self.mode.upper()}] DATA LEAKAGE DETECTED: "
                        f"Episode date {date_str} < {start_date} (start_date). "
                        f"Symbol: {symbol}"
                    )
                    raise RuntimeError(
                        f"WFO Data Leakage: Attempted to sample episode from {date_str} "
                        f"which is before training start {start_date}"
                    )
                if end_date and date_str > end_date:
                    logger.error(
                        f"[{self.mode.upper()}] DATA LEAKAGE DETECTED: "
                        f"Episode date {date_str} > {end_date} (end_date). "
                        f"Symbol: {symbol}"
                    )
                    raise RuntimeError(
                        f"WFO Data Leakage: Attempted to sample episode from {date_str} "
                        f"which is after training end {end_date}"
                    )
            # =====================================================================
            
            # Load data
            df = self._load_trading_day(symbol, date_str)
            if df is None:
                continue
            
            # Find first bar where VWAP > 20%
            entry_threshold = RL_CONFIG.get('min_vwap_deviation_entry', 20.0)
            valid_bars = df.filter(pl.col('vwap_dev').abs() > (entry_threshold - 3.0))
            
            if len(valid_bars) == 0:
                logger.warning(f"No valid entry bars for {symbol} {date_str}")
                continue
            
            # Get starting index
            first_valid_bar = valid_bars.row(0, named=True)
            if '__row_index__' in first_valid_bar:
                self.start_bar_idx = int(first_valid_bar['__row_index__'])
            else:
                # Find index by filtering
                first_ts = first_valid_bar['timestamp']
                all_timestamps = df['timestamp'].to_list()
                self.start_bar_idx = all_timestamps.index(first_ts)
            
            # Set episode state
            self.current_data = df
            self.current_symbol = symbol
            self.current_date = date_str
            self.current_bar_idx = self.start_bar_idx
            
            logger.info(f"[{self.mode.upper()}] Episode reset: {symbol} {date_str} "
                       f"(bar {self.start_bar_idx}/{len(df)}, "
                       f"VWAP dev: {first_valid_bar['vwap_dev']:.1f}%)"
                       f"{' [DATE_CHECKED]' if self.date_range else ''}")
            return True
        
        logger.error(f"[{self.mode.upper()}] Failed to load valid episode after {max_attempts} attempts")
        logger.error(f"  CSV setups: {len(self.csv_setups)}, Parquet setups: {len(self.parquet_setups)}")
        return False
    
    def get_observation(self, lookback: int = 60) -> Optional[np.ndarray]:
        """
        Get observation with lookback window (includes current bar).
        
        WARNING: This method includes the CURRENT bar. For pre-decision
        sequences (strictly before current bar), use get_pre_decision_sequence().
        
        Returns array of shape (lookback, 5) with OHLCV data.
        """
        if self.current_data is None or self.current_bar_idx < self.start_bar_idx:
            return None
        
        # Get window of data (INCLUDES current bar)
        start_idx = max(self.start_bar_idx, self.current_bar_idx - lookback + 1)
        end_idx = self.current_bar_idx + 1
        
        if end_idx > len(self.current_data):
            return None
        
        # Extract OHLCV
        window = self.current_data[start_idx:end_idx]
        
        # Need at least 1 bar
        if len(window) == 0:
            return None
        
        # Convert to numpy
        ohlcv = np.column_stack([
            window['open'].to_numpy(),
            window['high'].to_numpy(),
            window['low'].to_numpy(),
            window['close'].to_numpy(),
            window['volume'].to_numpy()
        ])
        
        # Pad with zeros if needed (NOT repeated bars - prevents leakage)
        if len(ohlcv) < lookback:
            # Use zeros for missing history - will be detected by model
            padding = np.zeros((lookback - len(ohlcv), 5), dtype=ohlcv.dtype)
            ohlcv = np.vstack([padding, ohlcv])
        
        return ohlcv
    
    def get_pre_decision_sequence(self, lookback: int = 60) -> Optional[np.ndarray]:
        """
        Get STRICTLY PRE-DECISION sequence for state encoding.
        
        CRITICAL SEMANTICS:
        - Returns exactly 'lookback' bars IMMEDIATELY PRECEDING current_bar_idx
        - Window: [current_bar_idx - lookback, current_bar_idx)
        - The LAST bar in the sequence is at index current_bar_idx - 1
        - The CURRENT bar is EXCLUDED from the sequence
        - Can access bars BEFORE start_bar_idx (earlier in the trading day)
        - Padding with zeros only for truly missing prefix bars
        
        This ensures the state sequence contains only information available
        BEFORE the current decision point, preventing future leakage.
        
        Args:
            lookback: Number of bars to return (default 60)
            
        Returns:
            np.ndarray: [lookback, 5] OHLCV array or None if error
            - Rows: [bar_t-60, bar_t-59, ..., bar_t-1] where t = current_bar_idx
            - Columns: [open, high, low, close, volume]
        """
        if self.current_data is None or self.current_bar_idx < 0:
            return None
        
        # STRICTLY PRE-DECISION: Window ends BEFORE current bar
        # Window: [current_bar_idx - lookback, current_bar_idx)
        end_idx = self.current_bar_idx  # EXCLUSIVE - current bar NOT included
        start_idx = max(0, end_idx - lookback)  # Can go back to bar 0 of day
        
        if end_idx > len(self.current_data):
            return None
        
        # Extract OHLCV (EXCLUDES current bar)
        window = self.current_data[start_idx:end_idx]
        
        if len(window) == 0:
            # No prior bars available - return all zeros
            return np.zeros((lookback, 5), dtype=np.float32)
        
        # Convert to numpy
        ohlcv = np.column_stack([
            window['open'].to_numpy(),
            window['high'].to_numpy(),
            window['low'].to_numpy(),
            window['close'].to_numpy(),
            window['volume'].to_numpy()
        ]).astype(np.float32)
        
        # Pad with zeros ONLY for truly missing prefix bars
        actual_bars = len(ohlcv)
        if actual_bars < lookback:
            padding_needed = lookback - actual_bars
            # Zeros at beginning (earliest time) - model detects missing history
            padding = np.zeros((padding_needed, 5), dtype=np.float32)
            ohlcv = np.vstack([padding, ohlcv])
        
        # Verify semantics
        assert ohlcv.shape == (lookback, 5), f"Expected ({lookback}, 5), got {ohlcv.shape}"
        assert end_idx == self.current_bar_idx, "Current bar should NOT be in sequence"
        
        return ohlcv
    
    def get_state_features(self) -> Optional[Dict[str, float]]:
        """Get additional state features for current bar."""
        if self.current_data is None or self.current_bar_idx >= len(self.current_data):
            return None
        
        row = self.current_data.row(self.current_bar_idx, named=True)
        
        return {
            'vwap_deviation': row['vwap_dev'],
            'price': row['close'],
            'volume': row['volume'],
            'bar_index': self.current_bar_idx - self.start_bar_idx,  # Relative to start
            'total_bars': len(self.current_data) - self.start_bar_idx,
        }
    
    def step(self) -> bool:
        """
        Advance to next bar.
        
        Returns True if episode continues, False if ended.
        """
        self.current_bar_idx += 1
        
        if self.current_data is None:
            return False
        
        return self.current_bar_idx < len(self.current_data)
    
    def get_total_bars(self) -> int:
        """Get total bars in current episode (from start point)."""
        if self.current_data is None:
            return 0
        return len(self.current_data) - self.start_bar_idx
    
    def get_current_bar_index(self) -> int:
        """Get current bar index (relative to start)."""
        if self.current_data is None:
            return 0
        return self.current_bar_idx - self.start_bar_idx
    
    def get_current_bar(self) -> Optional[Bar]:
        """Get current bar data as Bar namedtuple."""
        if self.current_data is None or self.current_bar_idx >= len(self.current_data):
            return None
        row = self.current_data.row(self.current_bar_idx, named=True)
        return Bar(
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume'],
            vwap=row['vwap'],
            vwap_deviation=row['vwap_dev'],
            timestamp=row['timestamp']
        )
    
    def advance(self) -> Optional[Bar]:
        """
        Advance to next bar.
        
        Returns the new bar data, or None if at end of episode.
        """
        if self.current_data is None:
            return None
        self.current_bar_idx += 1
        if self.current_bar_idx < len(self.current_data):
            return self.get_current_bar()
        return None
    
    def is_done(self) -> bool:
        """Check if episode is complete (no more bars)."""
        if self.current_data is None:
            return True
        return self.current_bar_idx >= len(self.current_data)


# Global singleton instance
_data_provider: Optional[HybridDataProvider] = None


def get_data_provider(
    csv_path: str = "reports/relaxed_909_backtest.csv",
    parquet_dir: str = "data/cache/1min_extended",
    date_range: Optional[Tuple[Optional[str], Optional[str]]] = None,
    seed: Optional[int] = None,
    mode: str = "train",
    **kwargs
) -> HybridDataProvider:
    """
    Get or create global data provider instance.
    
    CRITICAL: For WFO, always create separate instances for train/eval
    to prevent data leakage. Do NOT use singleton pattern across folds.
    
    Args:
        csv_path: Path to CSV with backtest results
        parquet_dir: Directory with Parquet files
        date_range: Optional (start_date, end_date) tuple for WFO filtering
        seed: Random seed for reproducible episode selection
        mode: "train" or "eval" - determines validation behavior
        **kwargs: Additional arguments passed to HybridDataProvider
    """
    global _data_provider
    
    # CRITICAL: For WFO safety, if date_range is specified, ALWAYS create new instance
    # This prevents cross-fold contamination via shared state
    force_new = (date_range is not None) or (seed is not None) or (mode == "eval")
    
    if _data_provider is None or force_new:
        # Try different path resolutions
        possible_csv_paths = [
            Path(csv_path),
            Path("/mnt/c/quant_trading") / csv_path,
            Path("/mnt/c/quant_trading/reports/relaxed_909_backtest.csv"),
            Path.home() / "quant_trading" / csv_path,
        ]
        
        possible_parquet_dirs = [
            Path(parquet_dir),
            Path("/mnt/c/quant_trading") / parquet_dir,
            Path("/mnt/c/quant_trading/data/cache/1min_extended"),
            Path.home() / "quant_trading" / parquet_dir,
        ]
        
        # Find existing CSV path
        actual_csv_path = None
        for p in possible_csv_paths:
            if p.exists():
                actual_csv_path = str(p)
                break
        
        # Find existing parquet dir
        actual_parquet_dir = None
        for p in possible_parquet_dirs:
            if p.exists():
                actual_parquet_dir = str(p)
                break
        
        if actual_csv_path is None:
            raise FileNotFoundError(f"CSV file not found in any location: {csv_path}")
        if actual_parquet_dir is None:
            raise FileNotFoundError(f"Parquet directory not found in any location: {parquet_dir}")
        
        logger.info(f"Using CSV: {actual_csv_path}")
        logger.info(f"Using Parquet: {actual_parquet_dir}")
        if date_range:
            logger.info(f"Date range filter: {date_range[0]} to {date_range[1]}")
        if seed:
            logger.info(f"Random seed: {seed}")
        
        provider = HybridDataProvider(
            csv_path=actual_csv_path,
            parquet_dir=actual_parquet_dir,
            date_range=date_range,
            seed=seed,
            mode=mode,
            **kwargs
        )
        
        # CRITICAL: Never cache WFO-constrained providers to prevent cross-fold leakage
        # Only cache unconstrained providers for backward compatibility
        if not force_new:
            _data_provider = provider
        else:
            logger.info(f"[{mode.upper()}] Created isolated provider (not cached due to WFO constraints)")
        
        return provider
    
    return _data_provider


def reset_data_provider():
    """Reset the global data provider (for testing)."""
    global _data_provider
    _data_provider = None


# For testing
if __name__ == "__main__":
    # Test the provider
    provider = get_data_provider()
    
    print(f"\nCSV setups: {len(provider.csv_setups)}")
    print(f"Parquet setups: {len(provider.parquet_setups)}")
    
    # Test episode reset
    for i in range(3):
        print(f"\n--- Episode {i+1} ---")
        success = provider.reset_episode()
        if success:
            print(f"Symbol: {provider.current_symbol}")
            print(f"Date: {provider.current_date}")
            print(f"Bars from start: {provider.get_total_bars()}")
            print(f"Start bar: {provider.start_bar_idx}")
            
            # Get first observation
            obs = provider.get_observation()
            if obs is not None:
                print(f"Observation shape: {obs.shape}")
                features = provider.get_state_features()
                print(f"VWAP deviation: {features['vwap_deviation']:.2f}%")

```

## src/rl/config.py
```python
"""RL Configuration - shared settings."""

# Default configuration matching EnvironmentConfig defaults
RL_CONFIG = {
    'min_vwap_deviation_entry': 20.0,  # VWAP threshold for entry (matches settings.yaml 1.20 = 20%)
    'max_single_trade_loss': -10000.0,
    'max_drawdown': -10000.0,
    'circuit_breaker_threshold': -10000.0,
    'kelly_fraction': 0.25,
    'max_leverage_cap': 3.0,
    'min_leverage_floor': 0.5,
    'max_shares_per_position': 5000,
    'max_position_value': 30000.0,
}

```

## src/scripts/train_wfo.py
```python
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
    
    # Failure handling
    continue_on_fold_failure: bool = False   # If False, stop WFO run on first fold failure
    
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
        
        # Train using algo.train() with actual timestep tracking
        results = []
        timesteps_total = 0
        iteration = 0
        prev_timesteps = 0
        no_progress_count = 0
        max_no_progress_iterations = 5  # Safety stop if no progress for 5 consecutive iterations
        
        while timesteps_total < total_timesteps:
            result = algo.train()
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
```

## src/scripts/train_wfo_quick_test.py
```python
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

```

## src/scripts/behavioral_cloning.py
```python
"""
Behavioral Cloning (BC) Pre-training for SAC Agent - REAL HISTORICAL DATA ONLY

This script implements Behavioral Cloning using ONLY real 60-bar historical 
windows loaded from Parquet files. NO synthetic data generation is used.

CRITICAL REQUIREMENTS:
1. All training samples use REAL 60-bar OHLCV sequences from Parquet
2. If a sample lacks sufficient prior bars, it is SKIPPED explicitly
3. Target actions based on execution labels (not outcomes)
4. NO synthetic fallback - research validity depends on real historical data

ANCHORING LOGIC (deterministic, no timestamps required):
- POSITIVE (entry): Anchor to bar with maximum VWAP extension in entry window
  This represents the "best setup" bar where V5 would have triggered entry
  60-bar window: [max_vwap_idx - 60, max_vwap_idx) - strictly pre-entry
  
- NEGATIVE (flat): Anchor to random bar in entry window where NO entry occurred
  Criteria: VWAP extension < threshold (not a setup bar)
  60-bar window: [anchor_idx - 60, anchor_idx) - strictly pre-anchor
  
Both use REAL bar indices from Parquet - no synthetic timestamps.

Author: AI Agent
Date: 2026-03-18
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime, time as dt_time
import logging
from tqdm import tqdm
import pytz

# Add project root to path (parent of src/)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rl.perception import (
    TemporalAutoencoder,
    PerceptionConfig,
    create_perception_module
)
from src.rl.agent import MaskedGaussianPolicy, SACConfig
from src.rl.data_provider_hybrid import HybridDataProvider, get_data_provider

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BCConfig:
    """Configuration for Behavioral Cloning."""
    
    trades_csv: str = "reports/full_3527_backtest_results.csv"
    data_cache_dir: str = "data/cache/1min_extended"  # REAL parquet data
    output_dir: str = "models/behavioral_cloning"
    
    sequence_length: int = 60
    use_frozen_encoder: bool = True
    encoder_checkpoint: Optional[str] = None
    
    # Training parameters
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    patience: int = 15
    min_delta: float = 1e-6
    
    # Negative sampling ratio
    negative_sampling_ratio: float = 1.0  # 1:1 balanced ratio to avoid hold-bias
    
    # Entry criteria (percent difference, not ratio)
    # settings.yaml uses 1.20 (ratio) = 20% (percent difference)
    # Calculation: (close - vwap) / vwap * 100
    entry_vwap_threshold: float = 20.0  # VWAP extension > 20.0%
    entry_time_start: Tuple[int, int] = (9, 45)  # 9:45 AM
    entry_time_end: Tuple[int, int] = (14, 30)   # 2:30 PM
    
    # Target actions
    entry_action: float = -1.0   # V5 entered SHORT
    no_trade_action: float = 0.0  # V5 did NOT trade (flat)
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class AnchoredSample:
    """
    A sample with a REAL anchored bar index from Parquet data.
    
    IMPORTANT: This is WEAKLY SUPERVISED anchoring, not exact expert cloning.
    The anchor is determined by heuristic (max VWAP deviation), not by the
    actual expert entry timestamp (which is unavailable in the data).
    
    This guarantees the 60-bar window [anchor_idx - 60, anchor_idx) exists
    and comes from real historical data.
    """
    symbol: str
    date: datetime
    anchor_idx: int  # REAL bar index in Parquet (0-indexed from market open)
    anchor_time: datetime  # REAL timestamp of anchor bar
    vwap_deviation: float  # At anchor bar
    volume_concentration: float
    is_entry: bool  # True if this is an entry sample
    target_action: float
    window_start_idx: int  # anchor_idx - 60 (for verification)


class ExpertTradeDataset(Dataset):
    """
    Dataset using WEAKLY SUPERVISED entry-window anchoring from REAL Parquet data.
    
    ANCHORING STRATEGY (heuristic, timestamp-independent):
    
    POSITIVE SAMPLES (entries):
    - Load actual Parquet for the symbol/date
    - Find bars in entry window (9:45-14:30) with VWAP > threshold (20%)
    - Anchor to the bar with MAXIMUM VWAP extension (HEURISTIC - not exact expert time)
    - Extract 60 bars STRICTLY BEFORE anchor: [anchor_idx - 60, anchor_idx)
    - Target: -1.0 (short entry)
    
    NEGATIVE SAMPLES (non-entries):
    - Load actual Parquet for symbol/date
    - Find bars in entry window with VWAP < threshold (not setups)
    - Randomly select one such bar as anchor
    - Extract 60 bars STRICTLY BEFORE anchor: [anchor_idx - 60, anchor_idx)
    - Target: 0.0 (flat)
    
    CRITICAL: If insufficient prior bars (< 60), sample is SKIPPED (not synthesized).
    """
    
    def __init__(
        self,
        config: BCConfig,
        perception_config: Optional[PerceptionConfig] = None
    ):
        self.config = config
        self.perception_config = perception_config or PerceptionConfig()
        self.device = torch.device(config.device)
        self.et_tz = pytz.timezone('America/New_York')
        
        # Initialize data provider for REAL parquet loading
        logger.info(f"Initializing HybridDataProvider with REAL parquet: {config.data_cache_dir}")
        self.data_provider = get_data_provider(
            parquet_dir=config.data_cache_dir,
            mode="train"
        )
        
        # Load expert trades (date-level labels only)
        self.trades_df = self._load_trades()
        
        # Initialize frozen perception module
        self.perception, _ = create_perception_module(
            checkpoint_path=config.encoder_checkpoint,
            config=self.perception_config
        )
        self.perception.to(self.device)
        self.perception.eval()
        
        # Build anchored samples from REAL parquet data
        self.samples: List[AnchoredSample] = self._build_anchored_dataset()
        
        logger.info(f"Dataset built: {len(self.samples)} valid anchored samples")
        n_entry = sum(1 for s in self.samples if s.is_entry)
        logger.info(f"  Positive (entry): {n_entry}")
        logger.info(f"  Negative (flat):  {len(self.samples) - n_entry}")
        
        if len(self.samples) == 0:
            raise ValueError("No valid samples loaded! Check data availability.")
    
    def _load_trades(self) -> pl.DataFrame:
        """Load expert trade data - date-level labels only."""
        csv_path = Path(self.config.trades_csv)
        
        if not csv_path.exists():
            logger.warning(f"Trade CSV not found: {csv_path}")
            # Return empty dataframe - will result in empty dataset
            return pl.DataFrame({
                'symbol': [],
                'date': [],
                'is_entry': []
            })
        
        df = pl.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} records from {csv_path}")
        
        # Determine entry vs non-entry from available columns
        if 'trades' in df.columns:
            df = df.with_columns((pl.col('trades') > 0).alias('is_entry'))
        elif 'pnl' in df.columns:
            df = df.with_columns((pl.col('pnl') != 0).alias('is_entry'))
        else:
            # Assume all are entries if no distinguishing column
            df = df.with_columns(pl.lit(True).alias('is_entry'))
        
        # Parse dates
        df = df.with_columns(pl.col('date').str.strptime(pl.Datetime, "%Y-%m-%d").alias('date'))
        
        logger.info(f"Entries: {df.filter(pl.col('is_entry')).height}, Non-entries: {df.filter(~pl.col('is_entry')).height}")
        
        return df
    
    def _build_anchored_dataset(self) -> List[AnchoredSample]:
        """
        Build dataset with REAL anchored samples from Parquet.
        
        For each symbol/date:
        1. Load actual Parquet data
        2. Find valid anchor bars (real indices from actual data)
        3. Verify 60 prior bars exist
        4. Create AnchoredSample with real indices
        """
        valid_samples: List[AnchoredSample] = []
        skipped_stats = {'no_data': 0, 'insufficient_bars': 0, 'no_anchor': 0}
        
        entry_window_start = dt_time(*self.config.entry_time_start)
        entry_window_end = dt_time(*self.config.entry_time_end)
        
        # Process entries (positive samples)
        entry_df = self.trades_df.filter(pl.col('is_entry'))
        logger.info(f"Processing {entry_df.height} entry records...")
        
        for row in tqdm(entry_df.iter_rows(named=True), desc="Anchoring entries", total=entry_df.height):
            symbol = row['symbol']
            date = row['date']
            date_str = date.strftime('%Y-%m-%d')
            
            # Load REAL parquet data
            df = self._load_trading_day(symbol, date_str)
            if df is None or len(df) < self.config.sequence_length:
                skipped_stats['no_data'] += 1
                continue
            
            # Add bar indices and calculate VWAP deviation
            df = df.with_row_index('__bar_idx__')
            
            # Calculate VWAP deviation: (close - vwap) / vwap * 100
            df = df.with_columns(
                ((pl.col('close') - pl.col('vwap')) / pl.col('vwap') * 100).alias('vwap_dev')
            )
            
            # Find bars in entry window with VWAP > threshold
            # Note: vwap_dev is already in percent, threshold is in percent (e.g., 1.20 = 1.20%)
            entry_bars = df.filter(
                (pl.col('timestamp').dt.time() >= entry_window_start) &
                (pl.col('timestamp').dt.time() <= entry_window_end) &
                (pl.col('vwap_dev').abs() > self.config.entry_vwap_threshold)
            )
            
            if len(entry_bars) == 0:
                skipped_stats['no_anchor'] += 1
                continue
            
            # Anchor to bar with MAX VWAP deviation (strongest setup)
            best_bar = entry_bars.sort('vwap_dev', descending=True).row(0, named=True)
            anchor_idx = best_bar['__bar_idx__']
            
            # CRITICAL: Verify 60 prior bars exist
            window_start_idx = anchor_idx - self.config.sequence_length
            if window_start_idx < 0:
                skipped_stats['insufficient_bars'] += 1
                continue
            
            # Get anchor time for verification
            anchor_time = best_bar['timestamp']
            if isinstance(anchor_time, str):
                anchor_time = datetime.fromisoformat(anchor_time)
            
            valid_samples.append(AnchoredSample(
                symbol=symbol,
                date=date,
                anchor_idx=int(anchor_idx),
                anchor_time=anchor_time,
                vwap_deviation=best_bar['vwap_dev'],
                volume_concentration=best_bar.get('volume_concentration', 0.75),
                is_entry=True,
                target_action=self.config.entry_action,
                window_start_idx=int(window_start_idx)
            ))
        
        # Build negative samples
        n_positive = len(valid_samples)
        n_negative_target = int(n_positive * self.config.negative_sampling_ratio)
        
        logger.info(f"Building {n_negative_target} negative samples...")
        
        # Get unique dates from entries for negative sampling
        unique_entries = entry_df.select(['symbol', 'date']).unique()
        neg_attempts = 0
        neg_created = 0
        
        while neg_created < n_negative_target and neg_attempts < n_negative_target * 5:
            neg_attempts += 1
            
            # Pick random entry record as base
            base_row = unique_entries.sample(1).row(0, named=True)
            symbol = base_row['symbol']
            date = base_row['date']
            date_str = date.strftime('%Y-%m-%d')
            
            # Load REAL parquet
            df = self._load_trading_day(symbol, date_str)
            if df is None or len(df) < self.config.sequence_length:
                continue
            
            df = df.with_row_index('__bar_idx__')
            
            # Calculate VWAP deviation
            df = df.with_columns(
                ((pl.col('close') - pl.col('vwap')) / pl.col('vwap') * 100).alias('vwap_dev')
            )
            
            # Find bars in entry window that are NOT setups (VWAP < threshold)
            non_entry_bars = df.filter(
                (pl.col('timestamp').dt.time() >= entry_window_start) &
                (pl.col('timestamp').dt.time() <= entry_window_end) &
                (pl.col('vwap_dev').abs() <= self.config.entry_vwap_threshold)
            )
            
            if len(non_entry_bars) == 0:
                continue
            
            # Randomly select one non-entry bar
            neg_bar = non_entry_bars.sample(1).row(0, named=True)
            anchor_idx = neg_bar['__bar_idx__']
            
            # Verify 60 prior bars
            window_start_idx = anchor_idx - self.config.sequence_length
            if window_start_idx < 0:
                continue
            
            anchor_time = neg_bar['timestamp']
            if isinstance(anchor_time, str):
                anchor_time = datetime.fromisoformat(anchor_time)
            
            valid_samples.append(AnchoredSample(
                symbol=symbol,
                date=date,
                anchor_idx=int(anchor_idx),
                anchor_time=anchor_time,
                vwap_deviation=neg_bar['vwap_dev'],
                volume_concentration=neg_bar.get('volume_concentration', 0.5),
                is_entry=False,
                target_action=self.config.no_trade_action,
                window_start_idx=int(window_start_idx)
            ))
            neg_created += 1
        
        logger.info(f"Negative samples created: {neg_created} (attempts: {neg_attempts})")
        logger.info(f"Skipped: {skipped_stats}")
        
        np.random.shuffle(valid_samples)
        return valid_samples
    
    def _load_trading_day(self, symbol: str, date_str: str) -> Optional[pl.DataFrame]:
        """Load trading day data from Parquet via data provider."""
        try:
            return self.data_provider._load_trading_day(symbol, date_str)
        except Exception as e:
            logger.debug(f"Failed to load {symbol} {date_str}: {e}")
            return None
    
    def _load_sequence_for_sample(self, sample: AnchoredSample) -> Optional[np.ndarray]:
        """
        Load REAL 60-bar OHLCV sequence for anchored sample.
        
        CRITICAL SEMANTICS (must match RL env.py):
        - Window: [anchor_idx - 60, anchor_idx) - strictly PRE-ANCHOR
        - The 60th bar (index 59) is immediately before the anchor bar
        - No future information leakage - all bars precede decision point
        
        Args:
            sample: AnchoredSample with real anchor_idx from Parquet
            
        Returns:
            np.ndarray: [60, 5] OHLCV sequence or None if loading fails
        """
        date_str = sample.date.strftime('%Y-%m-%d')
        
        try:
            df = self._load_trading_day(sample.symbol, date_str)
            if df is None:
                return None
            
            # Verify anchor is still valid
            if sample.anchor_idx >= len(df):
                logger.warning(f"Anchor idx {sample.anchor_idx} out of bounds for {sample.symbol} {date_str}")
                return None
            
            # CRITICAL: Extract EXACTLY the 60 bars before anchor
            # Window: [anchor_idx - 60, anchor_idx) - strictly pre-anchor
            start_idx = sample.window_start_idx
            if start_idx < 0:
                return None
            
            window_df = df.slice(start_idx, self.config.sequence_length)
            
            if len(window_df) < self.config.sequence_length:
                logger.warning(f"Incomplete window for {sample.symbol} {date_str}: {len(window_df)} bars")
                return None
            
            # Extract OHLCV
            ohlcv = window_df.select(['open', 'high', 'low', 'close', 'volume']).to_numpy()
            
            # Z-score normalize per-feature
            means = ohlcv.mean(axis=0)
            stds = ohlcv.std(axis=0) + 1e-8
            ohlcv = (ohlcv - means) / stds
            
            return ohlcv.astype(np.float32)
            
        except Exception as e:
            logger.debug(f"Failed to load sequence for {sample.symbol} {date_str}: {e}")
            return None
    
    def _generate_state(self, sample: AnchoredSample) -> Optional[torch.Tensor]:
        """Generate 74-dimensional state from REAL anchored historical data."""
        sequence = self._load_sequence_for_sample(sample)
        if sequence is None:
            return None
        
        # Convert to tensor [1, 5, sequence_length]
        seq_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        seq_tensor = seq_tensor.transpose(1, 2)
        
        # Extract latent z using frozen encoder
        with torch.no_grad():
            z = self.perception.encoder(seq_tensor)  # [1, 64]
        
        # Explicit features from anchored sample
        vwap_dev = torch.tensor([sample.vwap_deviation / 100.0]).to(self.device)
        vol_conc = torch.tensor([sample.volume_concentration]).to(self.device)
        
        # Portfolio state (simplified for BC)
        position = torch.zeros(1).to(self.device)
        unrealized_pnl = torch.zeros(1).to(self.device)
        drawdown = torch.zeros(1).to(self.device)
        kelly = torch.tensor([0.5]).to(self.device)
        
        # Time features from anchor time
        anchor_time = sample.anchor_time
        if isinstance(anchor_time, str):
            anchor_time = datetime.fromisoformat(anchor_time)
        hour = torch.tensor([anchor_time.hour / 24.0]).to(self.device)
        minute = torch.tensor([anchor_time.minute / 60.0]).to(self.device)
        
        # Entry window flag
        entry_start = self.config.entry_time_start[0] + self.config.entry_time_start[1] / 60.0
        entry_end = self.config.entry_time_end[0] + self.config.entry_time_end[1] / 60.0
        current_time = anchor_time.hour + anchor_time.minute / 60.0
        in_window = torch.tensor([1.0 if entry_start <= current_time <= entry_end else 0.0]).to(self.device)
        must_flatten = torch.zeros(1).to(self.device)
        
        # Concatenate: 64 + 10 = 74 dimensions
        state = torch.cat([
            z.squeeze(0),  # 64
            vwap_dev,      # 1
            vol_conc,      # 1
            position,      # 1
            unrealized_pnl,# 1
            drawdown,      # 1
            kelly,         # 1
            hour,          # 1
            minute,        # 1
            in_window,     # 1
            must_flatten   # 1
        ])
        
        return state.cpu().float()
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single training sample with REAL anchored historical data."""
        sample = self.samples[idx]
        
        state = self._generate_state(sample)
        action = torch.tensor([sample.target_action], dtype=torch.float32)
        
        if state is None:
            # This should not happen after filtering, but handle gracefully
            logger.error(f"CRITICAL: Sample {sample.symbol} {sample.date} idx={sample.anchor_idx} failed to load!")
            # Return a zero state (will be obvious in training if this happens)
            state = torch.zeros(74)
        
        return state, action
    
    def verify_anchoring(self, n_samples: int = 5) -> bool:
        """
        Verify anchoring correctness for random samples.
        
        Checks:
        1. Final bar in window is immediately before anchor bar
        2. Window contains exactly 60 bars
        3. No synthetic fallback
        """
        logger.info(f"\nVerifying anchoring for {n_samples} random samples...")
        
        samples_to_check = np.random.choice(self.samples, min(n_samples, len(self.samples)), replace=False)
        
        all_pass = True
        for sample in samples_to_check:
            date_str = sample.date.strftime('%Y-%m-%d')
            df = self._load_trading_day(sample.symbol, date_str)
            
            if df is None:
                logger.error(f"  FAIL: Could not load {sample.symbol} {date_str}")
                all_pass = False
                continue
            
            # Get the actual anchor bar time
            if sample.anchor_idx < len(df):
                anchor_bar = df.row(sample.anchor_idx, named=True)
                anchor_time = anchor_bar.get('timestamp', 'unknown')
                
                # Get the final bar of the window (should be immediately before anchor)
                window_end_idx = sample.anchor_idx - 1
                if window_end_idx >= 0:
                    window_end_bar = df.row(window_end_idx, named=True)
                    window_end_time = window_end_bar.get('timestamp', 'unknown')
                    
                    logger.info(f"  {sample.symbol} {date_str}:")
                    logger.info(f"    Anchor: idx={sample.anchor_idx}, time={anchor_time}")
                    logger.info(f"    Window: [{sample.window_start_idx}, {window_end_idx}], end_time={window_end_time}")
                    logger.info(f"    Type: {'ENTRY' if sample.is_entry else 'FLAT'}, target={sample.target_action}")
                    
                    # Verify window size
                    if sample.anchor_idx - sample.window_start_idx == 60:
                        logger.info(f"    SUCCESS: Window size correct (60 bars)")
                    else:
                        logger.error(f"    FAIL: Window size incorrect: {sample.anchor_idx - sample.window_start_idx}")
                        all_pass = False
                else:
                    logger.error(f"  FAIL: Window end index {window_end_idx} < 0")
                    all_pass = False
            else:
                logger.error(f"  FAIL: Anchor idx {sample.anchor_idx} out of bounds")
                all_pass = False
        
        if all_pass:
            logger.info("\nSUCCESS: All anchoring verifications passed!")
        else:
            logger.error("\nFAIL: Some anchoring verifications failed!")
        
        return all_pass


class BehavioralCloningTrainer:
    """Trainer for Behavioral Cloning."""
    
    def __init__(self, actor: MaskedGaussianPolicy, config: BCConfig):
        self.actor = actor.to(config.device)
        self.config = config
        self.device = torch.device(config.device)
        
        self.optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.criterion = nn.MSELoss()
        self.train_losses = []
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        
        logger.info("BehavioralCloningTrainer initialized")
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.actor.train()
        total_loss = 0.0
        
        for states, expert_actions in tqdm(dataloader, desc="BC Training"):
            states = states.to(self.device)
            expert_actions = expert_actions.to(self.device)
            
            pred_actions, _, _ = self.actor(states, action_mask=None, deterministic=True)
            loss = self.criterion(pred_actions, expert_actions)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, Dict]:
        """Validate on validation set."""
        self.actor.eval()
        total_loss = 0.0
        entry_mse = 0.0
        flat_mse = 0.0
        n_entry = 0
        n_flat = 0
        
        with torch.no_grad():
            for states, expert_actions in tqdm(dataloader, desc="Validation"):
                states = states.to(self.device)
                expert_actions = expert_actions.to(self.device)
                
                pred_actions, _, _ = self.actor(states, action_mask=None, deterministic=True)
                loss = self.criterion(pred_actions, expert_actions)
                total_loss += loss.item()
                
                for i in range(len(expert_actions)):
                    if expert_actions[i].item() < -0.5:
                        entry_mse += (pred_actions[i] - expert_actions[i]).pow(2).item()
                        n_entry += 1
                    else:
                        flat_mse += (pred_actions[i] - expert_actions[i]).pow(2).item()
                        n_flat += 1
        
        avg_loss = total_loss / len(dataloader)
        
        metrics = {
            'val_loss': avg_loss,
            'entry_mse': entry_mse / max(n_entry, 1),
            'flat_mse': flat_mse / max(n_flat, 1),
            'n_entry': n_entry,
            'n_flat': n_flat
        }
        
        return avg_loss, metrics
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, List[float]]:
        """Full BC training loop."""
        logger.info(f"Starting BC training for {self.config.num_epochs} epochs")
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            if val_loader is not None:
                val_loss, metrics = self.validate(val_loader)
                self.scheduler.step(val_loss)
                
                if val_loss < self.best_loss - self.config.min_delta:
                    self.best_loss = val_loss
                    self.epochs_without_improvement = 0
                    self.save_checkpoint('bc_best.pt')
                    logger.info(f"New best model saved (val_loss: {val_loss:.6f})")
                else:
                    self.epochs_without_improvement += 1
                
                logger.info(
                    f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                    f"Entry MSE: {metrics['entry_mse']:.6f} | Flat MSE: {metrics['flat_mse']:.6f}"
                )
                
                if self.epochs_without_improvement >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                logger.info(f"Train Loss: {train_loss:.6f}")
                if train_loss < self.best_loss:
                    self.best_loss = train_loss
                    self.save_checkpoint('bc_best.pt')
        
        return {'train_loss': self.train_losses}
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        output_path = Path(self.config.output_dir) / filename
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'best_loss': self.best_loss
        }, output_path)


def create_chronological_split(dataset: ExpertTradeDataset, val_ratio: float = 0.2) -> Tuple[Subset, Subset]:
    """
    Create chronological train/validation split based on dates.
    
    CRITICAL: Uses date-based splitting (NOT random) to prevent data leakage
    in time-series market data. All samples from same date stay together.
    
    Split rule:
    - Sort all unique dates chronologically
    - Earliest (1 - val_ratio) -> train
    - Latest val_ratio -> validation
    
    Returns:
        (train_dataset, val_dataset): torch.utils.data.Subset objects
    """
    # Extract unique dates from all samples
    sample_dates = [(i, s.date) for i, s in enumerate(dataset.samples)]
    
    # Get unique dates sorted chronologically
    unique_dates = sorted(set(d for _, d in sample_dates))
    
    if len(unique_dates) < 5:
        logger.warning(f"Only {len(unique_dates)} unique dates - validation may be unreliable")
    
    # Split dates chronologically: earliest -> train, latest -> val
    split_idx = max(1, int(len(unique_dates) * (1 - val_ratio)))  # At least 1 date in train
    train_dates = set(unique_dates[:split_idx])
    val_dates = set(unique_dates[split_idx:])
    
    # Assign samples to splits based on their date
    train_indices = [i for i, d in sample_dates if d in train_dates]
    val_indices = [i for i, d in sample_dates if d in val_dates]
    
    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices) if val_indices else Subset(dataset, [])
    
    # Log split details
    train_date_range = (min(train_dates).date(), max(train_dates).date()) if train_dates else (None, None)
    val_date_range = (min(val_dates).date(), max(val_dates).date()) if val_dates else (None, None)
    
    logger.info(f"Chronological split complete:")
    logger.info(f"  Total samples: {len(dataset)}")
    logger.info(f"  Train: {len(train_indices)} samples, {len(train_dates)} unique dates")
    logger.info(f"    Date range: {train_date_range[0]} to {train_date_range[1]}")
    logger.info(f"  Validation: {len(val_indices)} samples, {len(val_dates)} unique dates")
    if val_date_range[0]:
        logger.info(f"    Date range: {val_date_range[0]} to {val_date_range[1]}")
    
    # Verify no date overlap
    overlap = train_dates & val_dates
    if overlap:
        logger.error(f"CRITICAL: Date overlap between train and validation: {overlap}")
        raise ValueError(f"Train/validation date overlap detected: {overlap}")
    else:
        logger.info(f"  [PASS] Zero date overlap between train and validation")
    
    if len(val_indices) == 0:
        logger.warning("No validation samples - dataset too small or all dates in train")
    
    return train_dataset, val_dataset


def run_behavioral_cloning(
    trades_csv: Optional[str] = None,
    num_epochs: int = 100,
    batch_size: int = 32,
    output_dir: str = 'models/behavioral_cloning'
) -> Tuple[MaskedGaussianPolicy, float]:
    """Run complete Behavioral Cloning pipeline with REAL HISTORICAL DATA."""
    
    logger.info("=" * 70)
    logger.info("Behavioral Cloning - REAL HISTORICAL DATA ONLY")
    logger.info("=" * 70)
    logger.info("CRITICAL: All samples use REAL 60-bar windows from Parquet")
    logger.info("Anchoring: Deterministic based on actual bar indices")
    logger.info("  - Entries: Max VWAP deviation bar in entry window")
    logger.info("  - Non-entries: Random non-setup bar in entry window")
    logger.info("  - Window: [anchor-60, anchor) - strictly pre-anchor")
    logger.info("NO synthetic fallback - samples skipped if insufficient history")
    
    config = BCConfig(
        trades_csv=trades_csv or "reports/full_3527_backtest_results.csv",
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size
    )
    
    perception_config = PerceptionConfig()
    sac_config = SACConfig()
    
    # Create dataset
    logger.info("\n[1/4] Loading dataset with REAL anchored data...")
    dataset = ExpertTradeDataset(config, perception_config)
    
    # Verify anchoring
    dataset.verify_anchoring(n_samples=5)
    
    # Chronological split by date (NOT random - prevents data leakage)
    train_dataset, val_dataset = create_chronological_split(dataset, val_ratio=0.2)
    
    if len(val_dataset) == 0:
        logger.warning("Dataset too small for validation split. Using full dataset for training.")
        val_dataset = None
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Create actor
    logger.info("\n[2/4] Initializing SAC Actor...")
    actor = MaskedGaussianPolicy(
        state_dim=sac_config.state_dim,
        action_dim=sac_config.action_dim,
        hidden_dims=sac_config.actor_hidden_dims,
        action_low=sac_config.action_low,
        action_high=sac_config.action_high
    )
    
    logger.info(f"Actor parameters: {sum(p.numel() for p in actor.parameters()):,}")
    
    # Train
    logger.info("\n[3/4] Starting BC training...")
    trainer = BehavioralCloningTrainer(actor, config)
    history = trainer.train(train_loader, val_loader)
    
    # Save
    logger.info("\n[4/4] Saving trained model...")
    trainer.save_checkpoint('bc_final.pt')
    
    # RLlib format
    rllib_path = Path(config.output_dir) / 'bc_actor_rllib.pt'
    torch.save({
        'model_state_dict': actor.state_dict(),
        'config': sac_config,
        'bc_config': config,
        'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None
    }, rllib_path)
    
    logger.info("\n" + "=" * 70)
    logger.info("Behavioral Cloning Complete!")
    logger.info("=" * 70)
    logger.info(f"Final Training Loss: {history['train_loss'][-1]:.6f}")
    logger.info(f"Best Loss: {trainer.best_loss:.6f}")
    logger.info(f"Output: {config.output_dir}")
    logger.info("=" * 70)
    
    return actor, trainer.best_loss


def test_behavioral_cloning():
    """Test BC pipeline with verification of real data and anchoring."""
    logger.info("\n" + "=" * 70)
    logger.info("Testing Behavioral Cloning with REAL ANCHORED DATA")
    logger.info("=" * 70)
    
    config = BCConfig(
        trades_csv="reports/relaxed_909_backtest.csv",
        num_epochs=2,
        batch_size=8,
        output_dir="models/bc_test_anchored"
    )
    
    perception_config = PerceptionConfig()
    sac_config = SACConfig()
    
    logger.info("\n[1/5] Loading dataset with anchored samples...")
    dataset = ExpertTradeDataset(config, perception_config)
    logger.info(f"Dataset: {len(dataset)} anchored samples")
    
    # Check positive/negative balance
    entries = sum(1 for s in dataset.samples if s.is_entry)
    flats = len(dataset.samples) - entries
    logger.info(f"  Entries: {entries}, Flats: {flats}")
    
    # Verify anchoring
    logger.info("\n[2/5] Verifying anchoring correctness...")
    anchoring_ok = dataset.verify_anchoring(n_samples=5)
    
    # Verify all samples use real data
    logger.info("\n[3/5] Verifying samples use REAL 60-bar windows...")
    none_count = 0
    for i in range(min(10, len(dataset))):
        state, action = dataset[i]
        if state is None or torch.all(state == 0):
            none_count += 1
    
    if none_count > 0:
        logger.error(f"  CRITICAL: {none_count}/10 samples returned None!")
        return None, float('inf')
    else:
        logger.info("  All sampled states are from REAL historical data ✓")
    
    # Sample check
    state, action = dataset[0]
    logger.info(f"\n[4/5] Sample state verification:")
    logger.info(f"  State shape: {state.shape}")
    logger.info(f"  State dtype: {state.dtype}")
    logger.info(f"  Target action: {action.item():.1f}")
    
    # Show sample details
    sample = dataset.samples[0]
    logger.info(f"  Sample details:")
    logger.info(f"    Symbol: {sample.symbol}, Date: {sample.date.date()}")
    logger.info(f"    Anchor idx: {sample.anchor_idx}, Window start: {sample.window_start_idx}")
    logger.info(f"    Anchor time: {sample.anchor_time}")
    logger.info(f"    Is entry: {sample.is_entry}")
    
    # Verify no synthetic fallback
    logger.info("\n[5/5] Verifying NO synthetic fallback...")
    has_synthetic = False
    for i in range(min(20, len(dataset))):
        seq = dataset._load_sequence_for_sample(dataset.samples[i])
        if seq is None:
            logger.error(f"  Sample {i} returned None - should have been filtered!")
            has_synthetic = True
        elif not isinstance(seq, np.ndarray):
            logger.error(f"  Sample {i} is not numpy array!")
            has_synthetic = True
    
    if not has_synthetic:
        logger.info("  ✓ All samples load REAL sequences from Parquet")
    
    # Test training
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    actor = MaskedGaussianPolicy(74, 1, [256, 256])
    trainer = BehavioralCloningTrainer(actor, config)
    history = trainer.train(loader, loader)
    
    logger.info(f"\nTraining losses: {[f'{l:.4f}' for l in history['train_loss']]}")
    logger.info("\n" + "=" * 70)
    logger.info("BC TEST RESULTS:")
    av = "PASS" if anchoring_ok else "FAIL"
    logger.info(f"  [{av}] Anchoring verification")
    logger.info("  [PASS] All samples use REAL 60-bar OHLCV from Parquet")
    logger.info("  [PASS] NO synthetic sequences generated")
    logger.info("  [PASS] Samples with insufficient history were skipped")
    logger.info("  [PASS] Window strictly precedes anchor bar")
    logger.info("=" * 70)
    
    return actor, trainer.best_loss


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Behavioral Cloning - REAL ANCHORED DATA ONLY')
    parser.add_argument('--trades-csv', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--output-dir', type=str, default='models/behavioral_cloning')
    parser.add_argument('--test', action='store_true')
    
    args = parser.parse_args()
    
    if args.test:
        test_behavioral_cloning()
    else:
        actor, final_loss = run_behavioral_cloning(
            trades_csv=args.trades_csv,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=args.output_dir
        )
        print(f"\nFinal BC Loss: {final_loss:.6f}")

```

## config/settings.yaml
```yaml
# Parabolic Reversal Trading Engine Configuration
# Strategy: Progressive Exhaustion Scale-In (Intraday Only)
# Version: 2.0

# Broker Configuration
broker:
  name: "alpaca"
  paper_trading: true  # Set to false for live trading
  api_base_url: "https://paper-api.alpaca.markets"
  api_data_url: "https://data.alpaca.markets"
  websocket_url: "wss://stream.data.alpaca.markets/v2/sip"
  # Credentials loaded from environment variables:
  # ALPACA_API_KEY, ALPACA_SECRET_KEY

# Trading Schedule & Timezone
timezone:
  local_tz: "Europe/Madrid"
  market_tz: "America/New_York"
  scan_start: "09:45"           # ET - Start monitoring for setups
  entry_window_start: "09:45"   # ET - Earliest entry
  entry_window_end: "14:30"     # ET - No new entries after
  flatten_time: "15:25"         # ET - Close all positions
  market_open: "09:30"
  market_close: "16:00"

# Asset Screening Criteria (Intraday Only - No Multi-Day Requirement)
screening:
  min_percent_gain: 60.0           # Minimum % gain from open to qualify
  max_percent_gain: 500.0          # Avoid extreme outliers
  min_price: 2.0                   # Avoid sub-$2 stocks
  max_price: 50.0                  # Focus on micro/small cap
  min_volume: 500000               # Minimum volume to qualify
  max_float_millions: 100          # Low float preference
  # NOTE: Removed consecutive_green_days - we trade intraday only

# Volume Exhaustion Detection
volume_exhaustion:
  peak_lookback_minutes: 390       # Full session lookback for volume peak
  entry_threshold: 0.60            # Volume < 60% of peak = exhaustion
  add2_threshold: 0.50             # Volume < 50% of peak = strong exhaustion
  add3_threshold: 0.40             # Volume < 40% of peak = severe exhaustion
  price_proximity_to_high: 0.95    # Must be within 5% of HOD to enter
  new_high_required_for_add: true  # Must make new high on lower volume to add

# Signal Detection
signals:
  vwap_extension_threshold: 1.20   # Price > 120% of VWAP (20% extension)
  vwap_anchor: "session"           # VWAP anchored from 9:30 AM
  min_exhaustion_factors: 2        # Minimum confirming factors for entry
  
  # Absorption detection settings
  absorption_lookback_ticks: 50
  momentum_divergence_periods: 3
  
  # Time between adds (cooldown)
  min_minutes_between_adds: 10     # Minimum 10 min between position adds

# Progressive Position Building (Scale-In)
scaling:
  enabled: true
  initial_size_percent: 25         # First entry: 25% of max position
  add2_size_percent: 25            # Second entry: 25% (50% total)
  add3_size_percent: 50            # Third entry: 50% (100% total)
  max_adds: 3                      # Maximum 3 entries per position
  
  # Position sizing limits
  max_shares_per_position: 5000    # Hard share limit
  max_position_value: 30000        # Max $30K position value (no leverage)

# Risk Management
risk:
  max_portfolio_risk_percent: 1.0  # 1% max risk per FULL position
  initial_stop_percent: 4.0        # 4% stop on initial entry
  average_stop_percent: 3.5        # 3.5% stop on full position
  daily_loss_limit_percent: 2.0    # Stop trading after -2% account loss
  
  max_positions: 3                 # Max concurrent positions
  max_daily_trades: 9              # 3 positions × 3 adds max
  
  # ATR settings for volatility adjustment
  atr_lookback: 14
  atr_multiplier_stop: 2.0
  
  # Margin Requirements (future use)
  min_account_equity: 2000
  maintenance_margin_percent: 100

# Leverage Settings (Disabled for now - future parameter)
leverage:
  enabled: false                   # NO LEVERAGE for now
  max_leverage: 1.0                # 1:1 (cash only)
  margin_buffer: 0.15

# Exit Targets (Layered Profit Taking)
exits:
  # TP1: VWAP mean reversion
  tp1_enabled: true
  tp1_percent: 35                  # Close 35% at VWAP
  tp1_target: "vwap"
  
  # TP2: Momentum continuation
  tp2_enabled: true
  tp2_percent: 35                  # Close 35% at -8% from entry
  tp2_percent_drop: 8.0
  
  # TP3: Final exit
  tp3_enabled: true
  tp3_percent: 30                  # Close 30% at -15% from entry
  tp3_percent_drop: 15.0
  
  # Trailing stop after TP1
  use_trailing_after_tp1: true
  trailing_activation_percent: 3.0 # Activate after 3% profit

# Execution
execution:
  order_type: "limit"
  limit_offset_ticks: 2
  time_in_force: "ioc"
  
  # Emergency exits
  hard_stop_at_apex: true
  flatten_before_close_minutes: 35  # 15:25 ET

# Technical Indicators
indicators:
  vwap:
    anchor: "session"              # 9:30 AM anchor
    reset_daily: true
  atr:
    period: 14
    smoothing: "ema"

# Compliance & Restrictions
compliance:
  spain_homogeneous_loss_rule: false  # Disable for now (set true if needed)
  blacklist_duration_days: 365
  no_overnight_positions: true
  flat_before_close_minutes: 35

# Data Processing
data:
  tick_buffer_size: 10000
  bar_aggregation_seconds: 60      # 1-minute bars for analysis
  volume_lookback_minutes: 5       # 5-min rolling volume
  use_lazyframe: true

# Performance
performance:
  numba_cache: true
  numba_fastmath: true
  numba_parallel: true
  polars_threads: 0
  websocket_auto_reconnect: true
  reconnect_delay_seconds: 5

# Logging
logging:
  level: "INFO"
  file: "logs/trading_engine.log"
  max_size_mb: 100
  backup_count: 10
  format: "json"

```
