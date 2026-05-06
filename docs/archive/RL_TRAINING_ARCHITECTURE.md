# Parabolic Reversal RL Training Architecture

## Overview

This document describes the institutional-grade Reinforcement Learning (RL) training system for the Parabolic Reversal trading strategy. The system implements Soft Actor-Critic (SAC) with Behavioral Cloning initialization, two-phase Critic warm-up, and mathematically rigorous risk controls calibrated to the V5 Relaxed circuit breaker threshold of -$19,180.

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Cognitive Core: Soft Actor-Critic (SAC)](#cognitive-core-soft-actor-critic-sac)
3. [Two-Phase Training with Behavioral Cloning](#two-phase-training-with-behavioral-cloning)
4. [Data Layer: HybridDataProvider](#data-layer-hybriddataprovider)
5. [Environment Layer: ParabolicReversalEnv](#environment-layer-parabolicreversalev)
6. [Walk Forward Optimization (WFO)](#walk-forward-optimization-wfo)
7. [System Architecture: Bottom to Top](#system-architecture-bottom-to-top)
8. [System Architecture: Top to Bottom](#system-architecture-top-to-bottom)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         WFO ORCHESTRATION                                    │
│                        (train_wfo.py)                                        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  WFO Split N: Train [6mo] → Test [1mo] → Metrics                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RAY RLLIB SAC TRAINING LOOP                               │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Phase 1: Critic Warm-Up (20,000 steps)                            │    │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │    │
│  │  │    Actor    │ XX │   Critic 1  │◄──►│    Replay Buffer        │  │    │
│  │  │   FROZEN    │ XX │   (Q-Net)   │    │    (1M transitions)     │  │    │
│  │  │  requires_grad=False│├──────────┤    └─────────────────────────┘  │    │
│  │  │     LR=0.0  │ XX │   Critic 2  │                                  │    │
│  │  │             │ XX │   (Q-Net)   │                                  │    │
│  │  └─────────────┘    └─────────────┘                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Phase 2: Joint Fine-Tuning (60,000+ steps)                        │    │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │    │
│  │  │    Actor    │◄──►│   Critic 1  │◄──►│    Replay Buffer        │  │    │
│  │  │  UNFROZEN   │    │   (Q-Net)   │    │                         │  │    │
│  │  │   LR=3e-4   │    │   LR=3e-4   │    │    Entropy Temperature  │  │    │
│  │  │ (momentum   │    │ (momentum   │    │    α (auto-tuned)       │  │    │
│  │  │   flushed)  │    │   preserved)│    │                         │  │    │
│  │  └─────────────┘    └─────────────┘    └─────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PARABOLIC REVERSAL ENVIRONMENT                            │
│                       (ParabolicReversalEnv)                                 │
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐           │
│  │   State Builder  │  │  Action Masking  │  │  Reward Computer │           │
│  │  (TCN-AE +       │  │  (VWAP > 23%)    │  │  (Sortino +      │           │
│  │   Features)      │  │                  │  │   Tail-Risk)     │           │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘           │
│           │                     │                     │                      │
│           └─────────────────────┴─────────────────────┘                      │
│                                 │                                            │
│                                 ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              HybridDataProvider (Episode Management)                 │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │    │
│  │  │ CSV Setups   │  │ Parquet      │  │ Episode      │              │    │
│  │  │ (172 valid)  │  │ (optional)   │  │ Generator    │              │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Cognitive Core: Soft Actor-Critic (SAC)

### Why SAC (Not DQN)

The system uses **Soft Actor-Critic (SAC)** exclusively because:

1. **Continuous Action Space**: SAC natively supports the continuous action space `[-1.0, 1.0]` required for dynamic Quarter-Kelly fractional position sizing. DQN is limited to discrete actions.

2. **Maximum Entropy Framework**: SAC maximizes expected return while maximizing policy entropy, encouraging exploration and preventing premature convergence to suboptimal strategies.

3. **Off-Policy Stability**: SAC's twin Q-networks and entropy regularization provide stable learning for high-dimensional financial state spaces.

### SAC Configuration

```python
from ray.rllib.algorithms.sac import SACConfig

config = (
    SACConfig()
    .framework('torch')
    .training(
        # Twin Q-networks for stability
        q_model_config={
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
        },
        # Gaussian policy for continuous actions
        policy_model_config={
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
            "custom_model": MaskedGaussianPolicy,  # Custom model with action masking
        },
        # Learning rates
        lr=3e-4,                    # Default for both Actor and Critic
        tau=0.005,                  # Soft target network update coefficient
        initial_alpha=1.0,          # Initial entropy coefficient
        target_entropy='auto',      # Automatic entropy tuning
        # Replay buffer
        replay_buffer_config={
            "capacity": 1000000,    # 1M transitions
            "prioritized_replay": True,
            "prioritized_replay_alpha": 0.6,
            "prioritized_replay_beta": 0.4,
        },
        # Training batch
        train_batch_size=256,
        optimization_sequence=[1, 1],  # 1 Actor update per 1 Critic update
    )
    .resources(
        num_gpus=1 if torch.cuda.is_available() else 0,
        num_cpus_for_local_worker=4,
    )
)
```

### SAC Neural Network Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ACTOR NETWORK (Policy)                                                      │
│  Input: 74-dim state → Output: Mean & Log-Std for Gaussian policy           │
│                                                                              │
│  State [74] → FC(256, ReLU) → FC(256, ReLU) → FC(128, ReLU)                 │
│                                   ├─► Mean [1] → Tanh → [-1, 1]              │
│                                   └─► Log-Std [1] (clamped)                  │
│                                                                              │
│  Action = tanh(Mean + exp(Log-Std) * ε) where ε ~ N(0,1)                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  CRITIC NETWORKS (Twin Q-Networks) - Two identical networks                  │
│  Input: 74-dim state + 1-dim action → Output: Q-value                       │
│                                                                              │
│  State [74] ──┐                                                              │
│  Action [1] ──┼─► Concat [75] → FC(256, ReLU) → FC(256, ReLU) → Q-value [1] │
│               │                                                              │
│               └─► (Duplicated for Q1 and Q2, min(Q1, Q2) used for target)   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Action Interpretation

```python
# Continuous action from SAC policy
action: float ∈ [-1.0, 1.0]

# Discretized for execution
if action < -0.3:
    action_type = 0  # Entry/Add to short (intensity proportional to |action|)
elif action > 0.3:
    action_type = 1  # Cover shorts (intensity proportional to action)
else:
    action_type = 2  # Hold/Neutral

# Position sizing uses Quarter-Kelly formula scaled by action magnitude
# Shares = KellyFraction * Capital * |action| / (Price * RiskPerShare)
```

---

## Two-Phase Training with Behavioral Cloning

### Overview

The training implements a **Bi-Level Optimization** strategy:

1. **Phase 1 (Warm-Up)**: 20,000 timesteps with Actor frozen, only Critics learn value function
2. **Phase 2 (Fine-Tuning)**: Actor unfrozen with momentum buffers flushed, joint Actor-Critic optimization

This prevents the Actor from learning poor policies before the Critics have accurate value estimates.

### Phase 1: Critic Warm-Up (0-20,000 Steps)

```python
class WarmupCallback(DefaultCallbacks):
    """
    Phase 1: Actor is completely frozen using two-layer protection.
    """
    
    def _freeze_actor(self, policy: Policy):
        """
        LAYER 1: Hard Graph Freeze
        """
        model = policy.model
        
        # Freeze all parameters in action_model (Actor network)
        if hasattr(model, 'action_model'):
            for param in model.action_model.parameters():
                param.requires_grad = False  # Hard freeze - no gradients flow through
        
        # LAYER 2: Zero Learning Rate
        optimizers = policy.get_optimizers()
        for opt in optimizers:
            if self._is_actor_optimizer(opt, model):
                for param_group in opt.param_groups:
                    param_group['lr'] = 0.0  # Secondary protection
```

**Key Properties of Phase 1**:
- Actor receives **no gradient updates** (`requires_grad=False`)
- Actor optimizer **LR = 0.0** (redundant safety)
- Twin Critics learn state-value function from random policy exploration
- Behavioral Cloning weights (if available) provide initial Actor bias

### Phase 2: Joint Fine-Tuning (20,000+ Steps)

```python
    def _unfreeze_actor(self, policy: Policy):
        """
        Phase 2: Unfreeze Actor with CRITICAL momentum buffer flush.
        """
        model = policy.model
        optimizers = policy.get_optimizers()
        
        for i, opt in enumerate(optimizers):
            if self._is_actor_optimizer(opt, model):
                # CRITICAL STEP: Clear Adam's momentum state
                # Prevents garbage momentum from Phase 1 from exploding
                if len(opt.state) > 0:
                    opt.state.clear()  # Flush momentum buffers (exp_avg, exp_avg_sq)
                    logger.info(f"Cleared optimizer {i} state (momentum buffers flushed)")
                
                # Restore gradient flow
                if hasattr(model, 'action_model'):
                    for param in model.action_model.parameters():
                        param.requires_grad = True
                
                # Restore learning rate
                for param_group in opt.param_groups:
                    param_group['lr'] = 3e-4
```

**Why Momentum Buffer Flush is Critical**:

Adam optimizer maintains:
- `exp_avg`: Exponential moving average of gradients (momentum)
- `exp_avg_sq`: Exponential moving average of squared gradients (variance)

During Phase 1, these accumulate garbage values since Actor receives no meaningful gradients. If we don't clear them:

```
Update = LR * (momentum) / sqrt(variance)
         = 3e-4 * (garbage) / sqrt(garbage) 
         = EXPLOSIVE first update
```

By calling `opt.state.clear()`, we reset:
- `exp_avg` → 0
- `exp_avg_sq` → 0

Giving the Actor a clean slate for Phase 2 joint optimization.

### Behavioral Cloning (Optional Pre-Training)

```python
# behavioral_cloning.py

class BehavioralCloning:
    """
    Pre-train Actor network using expert demonstrations from CSV setups.
    """
    
    def train(self, csv_setups: List[Dict]):
        """
        Supervised learning on successful trades.
        
        For each profitable setup:
          - Input: Market state at entry (VWAP > 23%)
          - Target: Action = -1.0 (maximum short)
          - Loss: MSE(predicted_action, target_action)
        """
        for epoch in range(100):
            for setup in csv_setups:
                # Load market state at entry
                state = self.encode_state(setup)
                
                # Expert action: aggressive short (-1.0)
                target_action = torch.tensor([-1.0])
                
                # Forward pass
                predicted_action = self.actor(state)
                
                # MSE loss
                loss = F.mse_loss(predicted_action, target_action)
                
                # Backward
                loss.backward()
                self.optimizer.step()
        
        # Save checkpoint for Phase 1 initialization
        torch.save(self.actor.state_dict(), 'models/bc_checkpoint.pt')
```

### Complete Training Timeline

```
Timesteps: 0        5,000      10,000     15,000     20,000     25,000...
           │          │          │          │          │          │
           ▼          ▼          ▼          ▼          ▼          ▼
┌──────────────┬──────────┬──────────┬──────────┬──────────┬──────────────┐
│   PHASE 1    │  PHASE 1 │  PHASE 1 │  PHASE 1 │ TRANSITION│   PHASE 2   │
│   WARM-UP    │  WARM-UP │  WARM-UP │  WARM-UP │           │ FINE-TUNING │
│              │          │          │          │           │             │
│ Actor:       │ Actor:   │ Actor:   │ Actor:   │ Actor:    │ Actor:      │
│  FROZEN      │  FROZEN  │  FROZEN  │  FROZEN  │  MOMENTUM │  UNFROZEN   │
│  requires_grad=False│ requires_grad=False│ requires_grad=False│ requires_grad=False│   FLUSHED │  requires_grad=True│
│  LR=0.0      │  LR=0.0  │  LR=0.0  │  LR=0.0  │  LR→3e-4  │  LR=3e-4    │
│              │          │          │          │           │             │
│ Critics:     │ Critics: │ Critics: │ Critics: │ Critics:  │ Critics:    │
│  TRAINING    │  TRAINING│  TRAINING│  TRAINING│  TRAINING │  TRAINING   │
│  LR=3e-4     │  LR=3e-4 │  LR=3e-4 │  LR=3e-4 │  LR=3e-4  │  LR=3e-4    │
│              │          │          │          │           │             │
│ Exploration: │          │          │          │           │             │
│  ε=1.0 → 0.1 │          │          │          │           │ ε=0.1 → 0.02│
└──────────────┴──────────┴──────────┴──────────┴──────────┴──────────────┘
```

---

## Data Layer: HybridDataProvider

### Purpose

The `HybridDataProvider` manages episode generation from historical trading setups, ensuring each episode starts at a valid parabolic entry condition (VWAP deviation > 20%).

### VWAP Calculation (Anchored from Market Open)

The system **recalculates VWAP** from 9:30 AM ET for every episode (ignoring pre-calculated VWAP in Parquet files):

```python
def _calculate_vwap(self, df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate VWAP anchored from market open (9:30 AM ET).
    """
    # Convert timestamps to ET
    et_times = df['timestamp'].dt.convert_time_zone('America/New_York')
    hours = et_times.dt.hour().cast(pl.Int32).to_numpy()
    minutes = et_times.dt.minute().cast(pl.Int32).to_numpy()
    
    # Find bars after 9:30 AM (570 minutes from midnight)
    minutes_from_midnight = hours * 60 + minutes
    after_open_mask = minutes_from_midnight >= (9 * 60 + 30)
    
    # Typical price = (high + low + close) / 3
    typical_price = ((high + low + close) / 3).to_numpy()
    volume = df['volume'].to_numpy()
    close = df['close'].to_numpy()
    
    # Cumulative VWAP from market open
    cum_pv = 0.0
    cum_vol = 0.0
    vwap_values = []
    
    for i in range(len(df)):
        if after_open_mask[i]:
            cum_pv += typical_price[i] * volume[i]
            cum_vol += volume[i]
            vwap_values.append(cum_pv / cum_vol if cum_vol > 0 else close[i])
        else:
            vwap_values.append(close[i])  # Before market open
    
    return df.with_columns([pl.Series('vwap', vwap_values)])
```

### VWAP Deviation Formula

```
VWAP Deviation (%) = ((Close - VWAP) / VWAP) × 100

Example:
  Close = $12.00
  VWAP = $10.00
  Deviation = ((12 - 10) / 10) × 100 = 20%
```

### Episode Selection Criteria

| Criteria | Value | Rationale |
|----------|-------|-----------|
| Min VWAP Deviation | 23% | Strategy entry threshold |
| Episode Start | First bar > 20% | Ensure valid entry window |
| Market Hours | 9:30 AM - 4:00 PM ET | Valid trading session |
| Min Bars | 60 | TCN-AE sequence length |
| Data Sources | CSV (172 setups) | Proven winners with PnL > $100 |

### Episode Reset Flow

```python
def reset_episode(self) -> bool:
    """
    1. Randomly select from 172 CSV setups
    2. Load trading day from Parquet
    3. Calculate VWAP from 9:30 AM
    4. Find first bar where VWAP deviation > 20%
    5. Episode starts at that bar
    """
    setup = random.choice(self.csv_setups)  # 70% weight
    # Returns: {"symbol": "IGC", "date": "2020-08-12", "pnl": 5297.13}
    
    df = self._load_trading_day(setup.symbol, setup.date)
    # df columns: [timestamp, open, high, low, close, volume, vwap, vwap_dev]
    
    # Find first valid entry point
    valid_bars = df.filter(pl.col('vwap_dev').abs() > 20.0)
    first_valid = valid_bars.row(0, named=True)
    
    self.start_bar_idx = first_valid['__row_index__']
    self.current_bar_idx = self.start_bar_idx
    
    return True
```

---

## Environment Layer: ParabolicReversalEnv

### State Space (74 Dimensions)

```
┌─────────────────────────────────────────────────────────────────────┐
│  STATE VECTOR (74-dim)                                              │
├─────────────────────────────────────────────────────────────────────┤
│  [0:64]   TCN-AE Latent Encoding (60-bar OHLCV → 64-dim)           │
│  [64]     VWAP Deviation (%)                                        │
│  [65]     Volume Concentration                                      │
│  [66]     Current Position (normalized -1 to +1)                    │
│  [67]     Unrealized PnL (% of capital)                             │
│  [68]     Current Drawdown (% of capital)                           │
│  [69]     Kelly Fraction (Quarter-Kelly: 0.25 × full Kelly)         │
│  [70]     Hour (0-23)                                               │
│  [71]     Minute (0-59)                                             │
│  [72]     Day of Week (0-4)                                         │
│  [73]     Is Entry Window (9:45-14:30)                              │
└─────────────────────────────────────────────────────────────────────┘
```

### Action Masking (Hard Constraints)

```python
def _create_action_mask(self, action: np.ndarray) -> np.ndarray:
    """
    Mask invalid actions with catastrophic penalty.
    """
    mask = np.ones_like(action)
    
    # Constraint 1: Entry only when VWAP > 23%
    if self.vwap_deviation < self.config.min_vwap_deviation_entry:
        if action < -0.3:  # Attempting entry/add
            mask = np.zeros_like(action)
            self.mask_violation_penalty = -5.0
    
    # Constraint 2: Cannot exceed max position
    if abs(self.position_shares) >= self.config.max_shares_per_position:
        if action < -0.3:  # Attempting to add
            mask = np.zeros_like(action)
    
    # Constraint 3: Cannot cover if flat
    if self.position_shares == 0 and action > 0.3:
        mask = np.zeros_like(action)
    
    return mask
```

### Reward Function: Normalized Sortino with Scaled Drawdown Penalty

**CRITICAL**: All dollar values are normalized to neural-stable range [-10, +10] to prevent gradient explosion in SAC's MSE loss.

```python
def _calculate_normalized_reward(self) -> float:
    """
    Calculate reward with proper neural network normalization.
    
    Reward = Sortino_norm - Penalty_drawdown - Cost_slippage + PnL_norm
    
    All components scaled to [-10, +10] range for SAC stability.
    """
    # RAW DOLLAR VALUES (for reference)
    realized_pnl = getattr(self, '_last_trade_pnl', 0.0)
    total_pnl = realized_pnl + self.unrealized_pnl  # e.g., $5,000
    current_drawdown = self.current_drawdown        # e.g., -$8,000
    position_change = abs(self.current_position_value - self._prev_position_value)
    
    # === 1. SORTINO COMPONENT (Normalized to [-5, +5]) ===
    if len(self.daily_returns) > 1:
        returns = np.array(self.daily_returns)
        mean_return = np.mean(returns)
        downside_std = np.std(returns[returns < 0]) if len(returns[returns < 0]) > 0 else 1e-6
        sortino = (mean_return - self.config.risk_free_rate) / downside_std
    else:
        sortino = 0.0
    
    # Scale Sortino: typical range [-2, +3] → [-5, +5]
    sortino_component = np.clip(sortino * 1.5, -5.0, 5.0)
    
    # === 2. DRAWDOWN PENALTY (Normalized quadratic) ===
    # Circuit breaker: -$19,180 → Target penalty: -10.0
    # Scale: penalty = -(excess_drawdown / 4180)² × 10.0
    
    max_acceptable_dd = abs(self.config.max_acceptable_drawdown)  # $15,000
    current_dd = abs(current_drawdown)
    
    if current_dd > max_acceptable_dd:
        excess_dd = current_dd - max_acceptable_dd  # e.g., $4,180 at circuit breaker
        # Quadratic penalty on excess only
        # At circuit breaker: -(4180/4180)² × 10 = -10.0
        drawdown_penalty = -((excess_dd / 4180.0) ** 2) * 10.0
    else:
        drawdown_penalty = 0.0
    
    drawdown_penalty = np.clip(drawdown_penalty, -10.0, 0.0)
    
    # === 3. SLIPPAGE COST (100 bps = 1.0%) ===
    # Micro-cap reality: bid-ask spreads are massive
    slippage_cost = position_change * 0.01  # $30,000 × 0.01 = $300
    
    # Normalize: $300 / $3,000 max daily × 5.0 scale = 0.5
    max_daily_slippage = 3000.0  # 3 full turns at 1%
    slippage_penalty = -abs(slippage_cost) / max_daily_slippage * 5.0
    slippage_penalty = np.clip(slippage_penalty, -5.0, 0.0)
    
    # === 4. PNL COMPONENT (Normalized) ===
    # $20,000 max PnL → +10.0 reward
    pnl_component = (total_pnl / 20000.0) * 10.0  # Scale: 0.0005
    pnl_component = np.clip(pnl_component, -10.0, 10.0)
    
    # === TOTAL REWARD (Sum and clip) ===
    reward = sortino_component + drawdown_penalty + slippage_penalty + pnl_component
    reward = np.clip(reward, -10.0, 10.0)  # Final safety clip
    
    return float(reward)
```

### Why Normalization is Critical

**SAC uses MSE Loss**: `L = (Q_target - Q_current)²`

**Unnormalized (Explodes)**:
```
Raw drawdown: -$19,180
Raw penalty: -$9,302,710
MSE error: (9,302,710)² ≈ 8.65 × 10¹³  → Gradient explosion, NaN
```

**Normalized (Stable)**:
```
Normalized drawdown: -10.0
Normalized penalty: -10.0
MSE error: (10)² = 100  → Stable gradient flow
```

### Drawdown Penalty Scaling

```
Excess Drawdown = Current_DD - Max_Acceptable
                = $19,180 - $15,000
                = $4,180

Penalty Formula:
  Penalty = -(Excess / $4,180)² × 10.0

At circuit breaker:
  Penalty = -(4180 / 4180)² × 10.0 = -10.0

At $16,000 drawdown:
  Excess = $1,000
  Penalty = -(1000 / 4180)² × 10.0 = -0.57

This provides smooth quadratic penalty without gradient explosion.
```

### Quarter-Kelly Position Sizing

```python
def _calculate_position_size(self, action: float) -> int:
    """
    Dynamic position sizing using Quarter-Kelly Criterion.
    
    Kelly Fraction = (p × b - q) / b
    Where:
      p = win probability (from historical)
      q = loss probability = 1 - p
      b = avg win / avg loss (payoff ratio)
    
    Quarter-Kelly = 0.25 × Full Kelly (conservative)
    """
    # Historical statistics from backtest
    win_prob = self.kelly_stats['win_rate']  # ~0.79 for V5 Relaxed
    payoff_ratio = self.kelly_stats['avg_win'] / abs(self.kelly_stats['avg_loss'])
    
    # Full Kelly
    kelly_f = (win_prob * payoff_ratio - (1 - win_prob)) / payoff_ratio
    
    # Quarter-Kelly (conservative)
    kelly_fraction = 0.25 * kelly_f
    kelly_fraction = np.clip(kelly_fraction, 0.5, 3.0)  # Bounds
    
    # Scale by action intensity
    exposure = kelly_fraction * abs(action)
    
    # Convert to shares
    risk_per_share = self.current_price * 0.04  # 4% stop
    dollar_risk = self.capital * exposure * 0.01  # 1% capital risk
    shares = int(dollar_risk / risk_per_share)
    
    return min(shares, self.config.max_shares_per_position)
```

---

## Walk Forward Optimization (WFO)

### Purpose

WFO prevents overfitting by training on historical periods and testing on strictly future periods, simulating real deployment.

### Time Splits

```
Timeline: 2019 ────────────────────────────────────────────────► 2024

Fold 1:
  Train: 2019-01-01 ──► 2019-06-30  (6 months)
  Test:  2019-07-05 ──► 2019-08-04  (1 month forward)

Fold 2:
  Train: 2019-02-01 ──► 2019-07-31  (6 months, shifted 1 month)
  Test:  2019-08-05 ──► 2019-09-04  (1 month forward)
...

Fold 18 (Quick Test):
  Train: 2023-01-01 ──► 2023-06-30  (6 months)
  Test:  2023-07-05 ──► 2023-08-04  (1 month forward)
```

### Training Process per Fold

```python
def train_fold(self, fold: int, train_start, train_end, test_start, test_end):
    """
    Execute one WFO fold with full two-phase training.
    """
    # 1. Create SAC configuration
    config = self.create_sac_config(fold)
    
    # 2. Initialize algorithm
    algo = config.build()
    
    # 3. Phase 1: Critic Warm-Up (20,000 steps)
    #    - Actor frozen (requires_grad=False, LR=0.0)
    #    - Only Critics learn
    for step in range(20000):
        result = algo.train()
    
    # 4. Transition: Unfreeze Actor, flush momentum
    #    - opt.state.clear() called on Actor optimizer
    #    - requires_grad=True restored
    #    - LR restored to 3e-4
    
    # 5. Phase 2: Joint Fine-Tuning (60,000+ steps)
    #    - Both Actor and Critics train
    #    - Entropy temperature auto-tunes
    for step in range(20000, 80000):
        result = algo.train()
    
    # 6. Evaluation on test period (ε=0, greedy)
    eval_results = algo.evaluate()
    test_pnl = eval_results['episode_reward_mean']
    
    return {
        'fold': fold,
        'test_pnl': test_pnl,
        'win_rate': eval_results['custom_metrics']['win_rate'],
    }
```

---

## System Architecture: Bottom to Top

```
LAYER 1: RAW DATA
├── reports/relaxed_909_backtest.csv (909 setups, 172 with PnL > $100)
└── data/cache/1min_extended/*.parquet (3,089 symbols, 1-min OHLCV)

LAYER 2: INDEX BUILDING
├── Filter CSV: PnL > $100 → 172 setups
├── VWAP Validation: max deviation ≥ 23% (recalculated from 9:30 AM)
└── Cache: hybrid_index.pkl (172 valid episodes)

LAYER 3: EPISODE GENERATION
├── reset_episode(): Random sample from 172 setups
├── _load_trading_day(): Load Parquet, filter to date
├── _calculate_vwap(): Anchor from 9:30 AM ET
└── Find entry: First bar where VWAP > 20%

LAYER 4: ENVIRONMENT STEP
├── Action: Continuous [-1, 1] from SAC policy
├── Mask: Block if VWAP < 23% (penalty = -5)
├── Execute: Quarter-Kelly position sizing
├── Advance: Next bar, update PnL
├── Reward: Sortino - λ_dd(DD²) - Cost(Δw)
└── Return: (state[74], reward, done, info)

LAYER 5: SAC TRAINING
├── Phase 1 (0-20k): Actor frozen, Critics train
├── Transition: opt.state.clear() (momentum flush)
├── Phase 2 (20k+): Joint Actor-Critic training
└── Experience: Stored in 1M capacity replay buffer

LAYER 6: WFO ORCHESTRATION
├── Split: Train [6mo] → Test [1mo]
├── Aggregate: Mean PnL, Win Rate, Max DD
└── Report: Per-fold and aggregate metrics
```

---

## System Architecture: Top to Bottom

```
TRAINING SCRIPT (train_wfo_quick_test.py)
│
├── Configuration
│   ├── Algorithm: SAC (NOT DQN)
│   ├── Actor LR: 3e-4 (Phase 2), 0.0 (Phase 1)
│   ├── Critic LR: 3e-4 (constant)
│   ├── Warmup Steps: 20,000
│   └── Total Steps: 80,000
│
├── WFO Loop
│   └── For each time split:
│       ├── Build SAC config with WarmupCallback
│       ├── algo.train() → Phase 1 (Actor frozen)
│       ├── Transition (momentum flush)
│       ├── algo.train() → Phase 2 (joint)
│       └── Evaluate on test period
│
├── WarmupCallback Hooks
│   ├── on_algorithm_init():
│   │   ├── Load BC weights (if available)
│   │   └── _freeze_actor(): requires_grad=False, LR=0.0
│   │
│   └── on_train_result():
│       ├── Check timestep ≥ 20,000
│       └── _unfreeze_actor(): opt.state.clear(), LR=3e-4
│
└── Rollout Workers
    └── Each worker:
        ├── Creates ParabolicReversalEnv
        ├── env.reset() → HybridDataProvider.reset_episode()
        │   └── Returns episode starting at VWAP > 20%
        ├── env.step(action):
        │   ├── Apply action mask (VWAP > 23% check)
        │   ├── Quarter-Kelly sizing
        │   ├── Calculate reward:
        │   │   ├── Sortino component
        │   │   ├── Drawdown penalty (normalized to -10.0)
        │   │   └── Slippage cost (100bps = 1.0%)
        │   └── Return (state, reward, done)
        └── Collect transitions → Replay Buffer
```

---

## Key Mathematical Specifications

| Parameter | Value | Mathematical Rationale |
|-----------|-------|----------------------|
| **Algorithm** | SAC | Continuous actions for Kelly sizing |
| **Actor LR (Phase 1)** | 0.0 | Hard freeze via requires_grad=False |
| **Actor LR (Phase 2)** | 3e-4 | Standard SAC learning rate |
| **Warmup Steps** | 20,000 | Critics learn value function before Actor |
| **Drawdown Penalty** | Normalized quadratic | Max -10.0 at circuit breaker |
| **Circuit Breaker** | -$19,180 | V5 Relaxed empirical max drawdown |
| **MDD_max** | -$15,000 | Warning threshold for penalty activation |
| **Slippage** | 100 bps | Micro-cap bid-ask reality (0.01) |
| **Mask Penalty** | -5.0 | Stable gradient (not -1e9 explosion) |
| **Quarter-Kelly** | 0.25×Full | Conservative position sizing |

---

## Summary

The system implements institutional-grade RL training with:

1. **SAC Algorithm**: Continuous actions for dynamic Kelly sizing (not discrete DQN)
2. **Two-Phase Training**: 20k steps frozen Actor, momentum flush, then joint optimization
3. **Normalized Risk Control**: -10.0 penalty at circuit breaker (neural-stable scaling)
4. **Execution Friction**: 100bps (1.0%) slippage for micro-cap reality
5. **VWAP Anchoring**: All episodes start at valid entry points (>20% deviation)
6. **WFO Validation**: Strict train/test separation prevents overfitting
7. **Neural Normalization**: All dollar values scaled to [-10, +10] for SAC stability
