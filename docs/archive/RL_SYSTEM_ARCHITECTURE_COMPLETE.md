# Parabolic Reversal RL Trading System - Complete Architecture

## Executive Summary

This document describes the institutional-grade Reinforcement Learning (RL) trading system for the Parabolic Reversal strategy. The system implements Soft Actor-Critic (SAC) with mathematically rigorous risk controls, proper neural normalization, and realistic micro-cap market friction modeling.

**Key Differentiators:**
- SAC (not DQN) for continuous action space supporting dynamic Kelly sizing
- Two-phase training: 20k frozen Actor steps with momentum buffer flush
- Neural-normalized rewards [-30, +15] preventing gradient explosion
- 100 bps slippage modeling micro-cap reality
- Anchored VWAP calculation from 9:30 AM ET with look-ahead bias protection

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Data Layer](#data-layer)
3. [Perception Layer](#perception-layer)
4. [Environment Layer](#environment-layer)
5. [RL Training Layer](#rl-training-layer)
6. [Walk Forward Optimization](#walk-forward-optimization)
7. [Complete Data Flow](#complete-data-flow)
8. [Mathematical Specifications](#mathematical-specifications)
9. [Safety Mechanisms](#safety-mechanisms)

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         WFO ORCHESTRATION (train_wfo.py)                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │ WFO Split 1 │───►│ WFO Split 2 │───►│ WFO Split 3 │───►│     ...     │      │
│  │(6mo train/  │    │(6mo train/  │    │(6mo train/  │    │(6mo train/  │      │
│  │ 1mo test)   │    │ 1mo test)   │    │ 1mo test)   │    │ 1mo test)   │      │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘      │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         RAY RLLIB SAC TRAINING LOOP                              │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  PHASE 1: CRITIC WARM-UP (0-20,000 steps)                               │    │
│  │  ┌──────────────┐     ┌──────────────┐     ┌─────────────────────────┐  │    │
│  │  │     ACTOR    │     │   CRITIC 1   │◄────┤    REPLAY BUFFER        │  │    │
│  │  │   FROZEN     │     │   (Q-NET)    │     │    (1M transitions)     │  │    │
│  │  │ requires_grad=False  │   LR=3e-4    │◄───►│                         │  │    │
│  │  │     LR=0.0   │     │   MOMENTUM   │     │    Entropy Temp α       │  │    │
│  │  └──────────────┘     │   PRESERVED  │     │    (auto-tuned)         │  │    │
│  │                       └──────────────┘     └─────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  TRANSITION: MOMENTUM BUFFER FLUSH                                        │    │
│  │  opt.state.clear()  ──►  removes accumulated garbage from Phase 1        │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  PHASE 2: JOINT FINE-TUNING (20,000+ steps)                             │    │
│  │  ┌──────────────┐     ┌──────────────┐     ┌─────────────────────────┐  │    │
│  │  │   ACTOR      │◄───►│   CRITIC 1   │◄───►│    REPLAY BUFFER        │  │    │
│  │  │ UNFROZEN     │     │   (Q-NET)    │     │                         │  │    │
│  │  │ requires_grad=True   │   LR=3e-4    │     │    Standard SAC         │  │    │
│  │  │     LR=3e-4  │     │   MOMENTUM   │     │    operation            │  │    │
│  │  │ MOMENTUM=0   │     │   PRESERVED  │     │                         │  │    │
│  │  └──────────────┘     └──────────────┘     └─────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    PARABOLIC REVERSAL ENVIRONMENT                                │
│                       (ParabolicReversalEnv)                                     │
│                                                                                  │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────────┐     │
│  │   STATE BUILDER    │  │   ACTION MASKING   │  │   REWARD COMPUTER      │     │
│  │  (TCN-AE Encoder)  │  │  (VWAP > 23%)      │  │  (Normalized Sortino   │     │
│  │                    │  │                    │  │   + Drawdown + Slip)   │     │
│  └─────────┬──────────┘  └─────────┬──────────┘  └──────────┬─────────────┘     │
│            │                       │                        │                    │
│            └───────────────────────┼────────────────────────┘                    │
│                                    │                                             │
│                                    ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │              HYBRID DATA PROVIDER (Episode Management)                   │    │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │    │
│  │  │ CSV SETUPS      │  │ VWAP ANCHORED   │  │ FEATURE ENGINEERING     │  │    │
│  │  │ (172 validated  │──►│ CALCULATION     │──►│ - VWAP Deviation       │  │    │
│  │  │  with PnL>$100) │  │ (9:30 AM ET)    │  │ - Volume Concentration  │  │    │
│  │  └─────────────────┘  └─────────────────┘  │ - EMA 20-period         │  │    │
│  │                                             │ - No look-ahead bias    │  │    │
│  │                                             └─────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         RAW DATA SOURCES                                         │
│                                                                                  │
│  ┌──────────────────────────────┐  ┌─────────────────────────────────────────┐  │
│  │   reports/                   │  │   data/cache/1min_extended/             │  │
│  │   relaxed_909_backtest.csv   │  │   SYMBOL_1min_*.parquet                 │  │
│  │   - 909 trading setups       │  │   - 3,089 symbols                       │  │
│  │   - 172 with PnL > $100      │  │   - 1-minute OHLCV                      │  │
│  │   - VWAP validated           │  │   - Anchored VWAP calculated at load    │  │
│  └──────────────────────────────┘  └─────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Layer

### HybridDataProvider

**Purpose**: Bridge between raw Parquet files and RL environment, managing episode generation with proper VWAP anchoring and feature engineering.

**Data Flow**:
```
Raw Parquet (OHLCV)
       │
       ▼
┌─────────────────────────┐
│  _load_trading_day()    │
│  - Filter to date       │
│  - Sort by timestamp    │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  _engineer_features()   │
│  - Anchored VWAP calc   │
│  - VWAP deviation       │
│  - Volume concentration │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  MarketBar dataclass    │
│  (with all features)    │
└─────────────────────────┘
```

### Anchored VWAP Calculation

**Critical**: VWAP is recalculated from 9:30 AM ET for every episode (not using pre-calculated values from Parquet).

**Timezone Handling**: Raw Alpaca data is timezone-naive. Must localize to UTC first, then convert to ET.

```python
def _engineer_features(self, df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate VWAP anchored from market open (9:30 AM ET).
    
    TIMEZONE SAFETY: Alpaca raw data is timezone-naive.
    Chain: naive → UTC → America/New_York
    """
    # CRITICAL: Must localize naive data to UTC before converting to ET
    df = df.with_columns([
        pl.col('_ts')
        .dt.replace_time_zone('UTC')              # Step 1: Localize naive as UTC
        .dt.convert_time_zone('America/New_York') # Step 2: Convert to ET
        .dt.hour()
        .cast(pl.Int32)
        .alias('et_hour'),
        
        pl.col('_ts')
        .dt.replace_time_zone('UTC')              # Step 1: Localize naive as UTC  
        .dt.convert_time_zone('America/New_York') # Step 2: Convert to ET
        .dt.minute()
        .cast(pl.Int32)
        .alias('et_minute')
    ])
    
    # Identify bars after 9:30 AM (570 minutes from midnight)
    df = df.with_columns([
        ((pl.col('et_hour') * 60 + pl.col('et_minute')) >= 570)
        .alias('after_open')
    ])
    
    # Typical price = (high + low + close) / 3
    df = df.with_columns([
        ((pl.col('high') + pl.col('low') + pl.col('close')) / 3)
        .alias('typical_price')
    ])
    
    # Calculate VWAP from market open using numpy
    after_open = df['after_open'].to_numpy()
    typical_pv = (df['typical_price'] * df['volume']).to_numpy()
    volume = df['volume'].to_numpy()
    close = df['close'].to_numpy()
    
    cum_pv = 0.0
    cum_vol = 0.0
    vwap_values = []
    
    for i in range(len(df)):
        if after_open[i]:
            cum_pv += typical_pv[i]
            cum_vol += volume[i]
            # Epsilon prevents division by zero
            vwap_values.append(cum_pv / (cum_vol + 1e-8))
        else:
            vwap_values.append(close[i])
    
    df = df.with_columns([pl.Series('vwap', vwap_values)])
```

### Feature Engineering

**VWAP Deviation**:
```
VWAP Deviation % = ((Close - VWAP) / (VWAP + EPS)) × 100
Clipped to [-200%, +200%] to prevent extreme outliers
```

**Volume Concentration**:
```
Volume Concentration = Volume / (EMA(Volume, 20) + EPS)
Clipped to [0, 10] (10× normal volume = extreme capitulation)
```

**NaN Handling (No Look-Ahead Bias)**:
```python
# CRITICAL: No backward fill to prevent look-ahead bias
# EMA is NaN for first 19 bars (warmup period)
# Use forward fill (propagate valid past values)
# Then fill remaining with neutral baseline (1.0)
df = df.with_columns([
    pl.col('volume_concentration')
    .fill_null(strategy='forward')  # Only past → future, never future → past
    .fill_null(1.0)                  # Neutral if no valid data yet
])
```

### Data Column Handling

**Raw Alpaca Schema**: Parquet files contain `timestamp`, `open`, `high`, `low`, `close`, `volume`  
**Internal Processing**: Normalized to `_ts` column after timestamp detection and parsing  
**Feature Engineering**: Operates on `_ts` (datetime), `open`, `high`, `low`, `close`, `volume`

### MarketBar Dataclass

```python
@dataclass 
class MarketBar:
    """Single bar with engineered features."""
    timestamp: datetime      # Original timestamp from Alpaca
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float              # Anchored from 9:30 AM ET (calculated)
    vwap_deviation: float    # Percentage from VWAP (calculated)
    volume_concentration: float  # Relative to 20-bar EMA (calculated)
```

---

## Perception Layer

### Temporal Convolutional Autoencoder (TCN-AE)

**Purpose**: Compress 60-bar OHLCV history (300 dimensions) into 64-dim latent vector for state representation.

**Architecture**:
```
Input: [batch, 5 features, 60 timesteps]
           │
           ▼
┌─────────────────────────────┐
│  CausalConv1d(5→32, k=3)    │  Dilated convolution
│  + ReLU + Dropout(0.2)      │  Causal padding (no look-ahead)
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│  CausalConv1d(32→64, k=3)   │
│  + ReLU + Dropout(0.2)      │
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│  CausalConv1d(64→128, k=3)  │
│  + ReLU + Dropout(0.2)      │
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│  AdaptiveAvgPool1d(1)       │  Compress to single vector
│  + Linear(128→64)           │  Latent dimension
└──────────┬──────────────────┘
           │
           ▼
Latent: [batch, 64] ──► State concatenation
```

### StateRepresentation Module

Combines latent vector with explicit features:
```
State Vector (74 dimensions):
[0:64]   TCN-AE latent encoding
[64]     VWAP deviation (normalized)
[65]     Volume concentration
[66]     Position size (normalized)
[67]     Unrealized PnL %
[68]     Current drawdown %
[69]     Kelly leverage fraction
[70]     Hour (normalized)
[71]     Minute (normalized)
[72]     In entry window flag
[73]     Must flatten flag
```

---

## Environment Layer

### ParabolicReversalEnv

**State Space**: 74-dimensional continuous vector

**Action Space**: Continuous [-1.0, +1.0]
```
Action < -0.3: Entry/Add to short (intensity ∝ |action|)
-0.3 ≤ Action ≤ 0.3: Hold/Neutral
Action > 0.3: Cover shorts (intensity ∝ action)
```

### Action Masking (Hard Constraints)

```python
def _create_action_mask(self) -> np.ndarray:
    """
    Mask invalid actions to enforce strategy rules.
    """
    mask = np.ones(self.action_space.shape, dtype=np.float32)
    
    # Constraint 1: Entry only when VWAP > 23%
    if self.vwap_deviation < self.config.min_vwap_deviation_entry:
        mask[0] = 0  # Block entry
        self.mask_violation_penalty = -10.0
    
    # Constraint 2: Max position limit
    if abs(self.position_shares) >= self.config.max_shares_per_position:
        mask[1] = 0  # Block add
    
    # Constraint 3: Cannot cover if flat
    if self.position_shares == 0:
        mask[2] = 0  # Block cover
    
    return mask
```

### Reward Function (Normalized)

**Formula**:
```
Reward = Sortino_norm + Drawdown_penalty + Slippage_penalty + PnL_norm

All components individually bounded, sum varies naturally [-30, +15]
```

**Component 1: Sortino (Normalized to [-5, +5])**:
```python
if len(self.daily_returns) > 1:
    returns = np.array(self.daily_returns)
    mean_return = np.mean(returns)
    downside_std = np.std(returns[returns < 0])
    sortino = (mean_return - risk_free_rate) / downside_std
else:
    sortino = 0.0

sortino_component = np.clip(sortino * 1.5, -5.0, 5.0)
```

**Component 2: Drawdown Penalty (Normalized to [-10, 0])**:
```python
# Quadratic penalty on excess drawdown beyond $15k
# At circuit breaker ($19,180): penalty = -10.0
max_acceptable_dd = 15000.0
current_dd = abs(self.current_drawdown)

if current_dd > max_acceptable_dd:
    excess_dd = current_dd - max_acceptable_dd  # Max: $4,180
    max_excess = 19180.0 - 15000.0  # $4,180
    # -(excess / max_excess)² × 10.0
    drawdown_penalty = -((excess_dd / max_excess) ** 2) * 10.0
else:
    drawdown_penalty = 0.0

drawdown_penalty = np.clip(drawdown_penalty, -10.0, 0.0)
```

**Component 3: Slippage Penalty (100 bps = 1.0%)**:
```python
# Micro-cap reality: massive bid-ask spreads
position_change = abs(self.current_position_value - prev_position_value)
slippage_cost = position_change * 0.01  # 1.0% = 100 bps

# Normalize: max daily ~$3,000 (3 turns at 1% on $30k)
max_daily_slippage = 3000.0
slippage_penalty = -(slippage_cost / max_daily_slippage) * 5.0
slippage_penalty = np.clip(slippage_penalty, -5.0, 0.0)
```

**Component 4: PnL (Normalized to [-10, +10])**:
```python
# $20,000 PnL → +10.0 reward
pnl_component = (total_pnl / 20000.0) * 10.0
pnl_component = np.clip(pnl_component, -10.0, 10.0)
```

**Final Sum (No Total Clip)**:
```python
reward = sortino_component + drawdown_penalty + slippage_penalty + pnl_component
# Natural range: [-30, +15]
# No np.clip() on total - preserves gradient topography
return float(reward)
```

---

## RL Training Layer

### Why SAC (Not DQN)

| Feature | SAC | DQN |
|---------|-----|-----|
| Action Space | Continuous [-1, 1] | Discrete |
| Kelly Sizing | Native support | Not possible |
| Exploration | Entropy maximization | ε-greedy |
| Stability | Twin Q-networks | Single Q-network |

### Two-Phase Training

**Phase 1: Critic Warm-Up (0-20,000 steps)**
```python
# Actor is FROZEN
for param in model.action_model.parameters():
    param.requires_grad = False  # No gradient flow

for opt in optimizers:
    if is_actor_optimizer(opt):
        opt.param_groups[0]['lr'] = 0.0  # Zero learning rate

# Only Critics update
```

**Transition: Momentum Buffer Flush**
```python
def _unfreeze_actor(self, policy: Policy):
    """CRITICAL: Clear Adam's momentum before Phase 2."""
    for opt in optimizers:
        if is_actor_optimizer(opt):
            # STEP 1: Clear accumulated garbage from Phase 1
            if len(opt.state) > 0:
                opt.state.clear()  # Flush exp_avg, exp_avg_sq
            
            # STEP 2: Restore gradient flow
            for param in model.action_model.parameters():
                param.requires_grad = True
            
            # STEP 3: Restore learning rate
            opt.param_groups[0]['lr'] = 3e-4
```

**Phase 2: Joint Fine-Tuning (20,000+ steps)**
```python
# Both Actor and Critics update
# Standard SAC operation
# Entropy temperature auto-tunes
```

### SAC Configuration

```python
config = (
    SACConfig()
    .framework('torch')
    .training(
        # Twin Q-networks
        q_model_config={
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
        },
        # Gaussian policy
        policy_model_config={
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
        },
        # Learning rates
        lr=3e-4,
        tau=0.005,  # Soft target update
        initial_alpha=1.0,  # Entropy coefficient
        target_entropy='auto',  # Auto-tune
        # Replay buffer
        replay_buffer_config={
            "capacity": 1000000,
            "prioritized_replay": True,
            "prioritized_replay_alpha": 0.6,
        },
        train_batch_size=256,
    )
)
```

---

## Walk Forward Optimization

### Purpose
Prevent overfitting by training on historical data and testing on strictly future data.

### Time Splits
```
Fold 1:  Train 2019-01-01 → 2019-06-30 | Test 2019-07-05 → 2019-08-04
Fold 2:  Train 2019-02-01 → 2019-07-31 | Test 2019-08-05 → 2019-09-04
Fold 3:  Train 2019-03-01 → 2019-08-31 | Test 2019-09-05 → 2019-10-04
...
Fold 18: Train 2023-01-01 → 2023-06-30 | Test 2023-07-05 → 2023-08-04
```

### Process
```python
for fold in wfo_splits:
    # 1. Create fresh SAC algorithm
    algo = config.build()
    
    # 2. Phase 1: Critic warm-up (20k steps, frozen Actor)
    for step in range(20000):
        result = algo.train()
    
    # 3. Transition: Unfreeze Actor, flush momentum
    callback.unfreeze_actor()
    
    # 4. Phase 2: Joint training (60k+ steps)
    for step in range(20000, 80000):
        result = algo.train()
    
    # 5. Evaluate on test period (ε=0, greedy)
    test_reward = algo.evaluate()
    
    # 6. Log results
    results.append({
        'fold': fold,
        'test_pnl': test_reward,
        'win_rate': calculate_win_rate(),
    })
```

---

## Complete Data Flow

### Episode Lifecycle

```
1. RESET EPISODE
   ├── DataProvider.reset_episode()
   │   ├── Random sample from 172 CSV setups
   │   ├── Load Parquet file for (symbol, date)
   │   ├── _engineer_features():
   │   │   ├── Calculate VWAP from 9:30 AM
   │   │   ├── VWAP deviation
   │   │   ├── Volume concentration (EMA 20)
   │   │   └── Safety nets (EPS, no NaN, clip)
   │   └── Find first bar where VWAP > 20%
   └── Return TradingDayData with MarketBar sequence

2. ENVIRONMENT STEP (for each bar)
   ├── Agent selects action: continuous [-1, 1]
   ├── Apply action mask (VWAP > 23% check)
   ├── If entry/add:
   │   ├── Quarter-Kelly position sizing
   │   └── Deduct 100 bps slippage
   ├── Advance to next bar
   ├── Update PnL
   ├── Build observation:
   │   ├── TCN-AE encode 60-bar history → 64-dim
   │   └── Concatenate explicit features → 74-dim
   ├── Calculate reward:
   │   ├── Sortino [-5, +5]
   │   ├── Drawdown penalty [-10, 0]
   │   ├── Slippage penalty [-5, 0]
   │   └── PnL [-10, +10]
   │   └── Sum naturally [-30, +15]
   └── Return (obs[74], reward, done, info)

3. SAC TRAINING
   ├── Store transition in replay buffer
   ├── Sample batch of 256 transitions
   ├── Update Twin Critics (MSE loss)
   ├── Update Actor (entropy-regularized)
   └── Soft update target networks

4. TERMINATION (when any)
   ├── End of trading day
   ├── Circuit breaker (-$19,180)
   └── Max steps reached
```

---

## Mathematical Specifications

| Parameter | Value | Mathematical Purpose |
|-----------|-------|---------------------|
| **Algorithm** | SAC | Continuous actions for Kelly sizing |
| **Phase 1 Steps** | 20,000 | Critic learns value function |
| **Phase 2 Steps** | 60,000+ | Joint Actor-Critic optimization |
| **Actor LR (P1)** | 0.0 | Hard freeze (requires_grad=False) |
| **Actor LR (P2)** | 3e-4 | Standard SAC learning rate |
| **Replay Buffer** | 1,000,000 | Large capacity for diverse experience |
| **Batch Size** | 256 | Stable gradient estimates |
| **Tau** | 0.005 | Soft target network update |
| **Slippage** | 100 bps | Micro-cap reality (1.0%) |
| **Sortino Scale** | 1.5× | Maps to [-5, +5] |
| **Drawdown Scale** | Quadratic | -10.0 at circuit breaker |
| **PnL Scale** | 0.0005 | $20k → 10.0 |
| **Reward Range** | [-30, +15] | Natural sum, no total clip |
| **VWAP Anchor** | 9:30 AM ET | Market open calculation |
| **EMA Period** | 20 bars | Volume concentration baseline |
| **Safety EPS** | 1e-8 | Division by zero protection |
| **Max VWAP Dev** | ±200% | Outlier clipping |
| **Max Vol Conc** | 10.0 | Extreme volume clipping |

---

## Safety Mechanisms

### 1. Computational Safety (NaN/Inf Prevention)
```python
# Epsilon for all divisions
vwap_deviation = (close - vwap) / (vwap + EPS) * 100
volume_conc = volume / (ema_volume + EPS)

# NaN handling (no look-ahead bias)
# Forward fill only, never backward
volume_conc = volume_conc.fill_null(strategy='forward').fill_null(1.0)

# Final numpy safety net
vwap_dev_np = np.nan_to_num(vwap_dev_np, nan=0.0, posinf=200.0, neginf=-200.0)

# Assertions (fail-fast)
assert not df['vwap_deviation'].is_null().any()
```

### 2. Risk Management
```python
# Circuit breaker at V5 Relaxed max drawdown
if self.current_drawdown <= -19180.0:
    self.circuit_breaker_triggered = True
    done = True

# Quarter-Kelly position sizing (conservative)
kelly_fraction = 0.25 * full_kelly
position_size = kelly_fraction * capital / (price * risk_per_share)

# Action masking (hard constraints)
if vwap_deviation < 23.0:
    mask[entry_action] = 0  # Block entry
```

### 3. Gradient Stability
```python
# Neural normalization (all values to [-10, +10] range)
# Prevents MSE loss explosion in SAC

# Individual component clipping (preserves gradient flow)
# No total reward clipping (distinguishes trade quality)

# Momentum buffer flush between phases
# Prevents garbage gradients from Phase 1 affecting Phase 2
```

---

## Summary

This architecture implements a production-grade RL trading system with:

1. **Proper Algorithm**: SAC for continuous action space (not DQN)
2. **Two-Phase Training**: Frozen Actor warmup with momentum flush
3. **Neural Safety**: All values normalized to prevent gradient explosion
4. **Market Reality**: 100 bps slippage modeling micro-cap friction
5. **No Look-Ahead**: VWAP anchored from 9:30 AM, forward-fill only
6. **Risk Management**: Circuit breaker at -$19,180, Quarter-Kelly sizing
7. **Computational Safety**: EPS protection, NaN handling, outlier clipping

The system is mathematically rigorous and ready for WFO training.
