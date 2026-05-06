# Reward Function and Slippage - Corrected Specification

## Critical Corrections

### 1. Slippage Model: 100 bps (1.0%) Micro-Cap Reality

**Previous Error**: 1 bps ($0.01\%$) slippage is detached from parabolic micro-cap reality.

**Corrected Implementation**:

```python
@dataclass
class EnvironmentConfig:
    """Corrected configuration for micro-cap execution friction."""
    
    # SLIPPAGE: 100 bps (1.0%) for parabolic micro-cap stocks
    # Rationale: Bid-ask spreads on volatile tickers (IGC, GTEC, etc.) 
    # routinely exceed 50-200 bps during parabolic moves
    transaction_cost_per_dollar: float = 0.01  # 100 bps = 1.0%
    
    # Example: $30,000 position change costs $300 in slippage
    # This prevents the agent from overtrading in simulation
```

**Slippage Calculation**:

```python
def _calculate_transaction_cost(self, position_change: float) -> float:
    """
    Micro-cap execution friction: 100 bps per trade.
    
    Args:
        position_change: Absolute dollar value of position change
        
    Returns:
        Slippage cost in dollars
    """
    return position_change * self.config.transaction_cost_per_dollar

# Example:
# Position change: $30,000 (entering full position)
# Slippage: $30,000 × 0.01 = $300
# 
# Position change: $15,000 (adding to position)
# Slippage: $15,000 × 0.01 = $150
```

### 2. Neural Network Normalization (Critical)

**The Problem**: Raw dollar values ($19,180 drawdown) cause gradient explosions in SAC.

**SAC uses MSE Loss**:
```
L_critic = (Q_target - Q_current)²
```

If Q-values are in range [-10,000,000, +10,000,000], MSE produces:
- Squared error: (10⁷)² = 10¹⁴
- Float32 max: ~3.4 × 10³⁸
- Gradient scale: Unstable, NaN propagation

**The Solution**: Normalize all dollar values to neural-stable range [-10, +10].

```python
@dataclass
class RewardScalingConfig:
    """Normalization constants for neural network stability."""
    
    # Maximum expected values (for normalization)
    MAX_DRAWDOWN_DOLLARS: float = 19180.0      # Circuit breaker
    MAX_PNL_DOLLARS: float = 20000.0           # Max daily PnL
    MAX_POSITION_VALUE: float = 30000.0        # Position limit
    
    # Neural network target range
    NN_MAX_PENALTY: float = -10.0              # Max drawdown penalty
    NN_MAX_REWARD: float = +10.0               # Max positive reward
    
    # Derived scaling factors
    DRAWDOWN_SCALE: float = NN_MAX_PENALTY / MAX_DRAWDOWN_DOLLARS  # ≈ -0.000521
    PNL_SCALE: float = NN_MAX_REWARD / MAX_PNL_DOLLARS             # ≈ 0.0005
```

### 3. Normalized Reward Function

```python
def _calculate_normalized_reward(self) -> float:
    """
    Calculate reward with proper neural network normalization.
    
    All dollar values are scaled to range [-10, +10] to prevent
    gradient explosion in SAC's MSE loss.
    
    Returns:
        Normalized reward in range [-10, 10]
    """
    # RAW DOLLAR VALUES
    realized_pnl = getattr(self, '_last_trade_pnl', 0.0)
    total_pnl = realized_pnl + self.unrealized_pnl
    current_drawdown = self.current_drawdown
    position_change = abs(self.current_position_value - self._prev_position_value)
    
    # === 1. SORTINO COMPONENT (Normalized) ===
    if len(self.daily_returns) > 1:
        returns = np.array(self.daily_returns)
        mean_return = np.mean(returns)
        negative_returns = returns[returns < 0]
        downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 1e-6
        sortino = (mean_return - self.config.risk_free_rate) / downside_std
    else:
        sortino = 0.0
    
    # Scale Sortino to [-5, +5] range
    # Typical Sortino values: -2 to +3
    # Scaling: Sortino × 1.0 → [-2, +3]
    sortino_component = np.clip(sortino * 1.0, -5.0, 5.0)
    
    # === 2. DRAWDOWN PENALTY (Normalized) ===
    # Raw drawdown: -$19,180 (circuit breaker)
    # Target penalty: -10.0
    # Scale factor: -10.0 / 19180 ≈ -0.000521
    
    max_acceptable_dd = abs(self.config.max_acceptable_drawdown)  # $15,000
    current_dd = abs(current_drawdown)
    
    if current_dd > max_acceptable_dd:
        # Excess drawdown beyond acceptable threshold
        excess_dd = current_dd - max_acceptable_dd  # e.g., $4,180 at circuit breaker
        
        # Quadratic penalty on EXCESS only
        # At circuit breaker: excess = $4,180
        # Raw penalty: (4180)² = 17,472,400
        # Normalized: 17,472,400 × (-0.000521) ≈ -9,103
        # Clipped to: -10.0
        
        raw_penalty = -(excess_dd ** 2)
        drawdown_penalty = raw_penalty * (10.0 / (19180 - 15000) ** 2)
        # Formula: penalty = -(excess²) × (10.0 / 4180²)
        # At max: -(4180²) × (10.0 / 4180²) = -10.0
    else:
        drawdown_penalty = 0.0
    
    drawdown_penalty = np.clip(drawdown_penalty, -10.0, 0.0)
    
    # === 3. TRANSACTION COST (Normalized) ===
    # 100 bps = 1.0% slippage
    slippage_cost = position_change * 0.01  # e.g., $30,000 × 0.01 = $300
    
    # Normalize: $300 / $3000 (max daily cost) × 1.0 = 0.1
    max_daily_slippage = 3000.0  # 3 full position turns at 1%
    slippage_penalty = -abs(slippage_cost) / max_daily_slippage  # Range: [0, -1.0]
    
    # === 4. PNL COMPONENT (Normalized) ===
    # Raw PnL: $20,000 target max
    # Normalized: $20,000 × 0.0005 = 10.0
    pnl_component = total_pnl * (10.0 / 20000.0)  # Scale factor: 0.0005
    pnl_component = np.clip(pnl_component, -10.0, 10.0)
    
    # === TOTAL REWARD (Clipped to safe range) ===
    reward = sortino_component + drawdown_penalty + slippage_penalty + pnl_component
    reward = np.clip(reward, -10.0, 10.0)
    
    return float(reward)
```

### 4. Why This Normalization Works

**Before (Unnormalized)**:
```
Raw Values:
  Drawdown: -$19,180
  Penalty: -9,302,710 (λ_dd = 5,688,000)
  
SAC Critic MSE:
  L = (Q_target - Q_current)²
  If Q_current = -9,302,710 and Q_target = -9,000,000
  L = (-302,710)² ≈ 9.16 × 10¹⁰
  
Gradient:
  ∂L/∂θ = 2 × error × ∂Q/∂θ
        = 2 × (-302,710) × (some gradient)
        = EXPLOSIVE (NaN in float32)
```

**After (Normalized)**:
```
Normalized Values:
  Drawdown: -$19,180 → -10.0 (after normalization)
  Penalty: -10.0 (scaled)
  
SAC Critic MSE:
  L = (Q_target - Q_current)²
  If Q_current = -10.0 and Q_target = -9.5
  L = (-0.5)² = 0.25
  
Gradient:
  ∂L/∂θ = 2 × (-0.5) × ∂Q/∂θ
        = -1.0 × ∂Q/∂θ
        = STABLE and well-conditioned
```

### 5. Complete Corrected Environment Config

```python
@dataclass
class EnvironmentConfig:
    """Production-grade configuration with neural-stable normalization."""
    
    # Circuit breaker from V5 Relaxed empirical data
    circuit_breaker_threshold: float = -19180.0
    max_drawdown: float = -19180.0
    
    # Reward normalization targets
    nn_max_penalty: float = -10.0    # Max network penalty
    nn_max_reward: float = +10.0     # Max network reward
    
    # Drawdown penalty (NORMALIZED)
    # At circuit breaker (-$19,180): penalty = -10.0
    # Formula: penalty = -(excess_drawdown / 4180)² × 10.0
    drawdown_penalty_scale: float = 10.0  # Max penalty magnitude
    
    # Slippage: 100 bps (1.0%) for micro-cap reality
    transaction_cost_per_dollar: float = 0.01  # 100 bps
    
    # PnL scaling: $20,000 PnL → +10.0 reward
    pnl_scale: float = 0.0005  # 10.0 / 20000
    
    # Masking penalty (stable gradient)
    masking_penalty: float = -5.0
```

### 6. Summary Table

| Parameter | Old (Incorrect) | New (Corrected) | Rationale |
|-----------|----------------|-----------------|-----------|
| **Slippage** | 1 bps (0.01%) | 100 bps (1.0%) | Micro-cap bid-ask reality |
| **Max Penalty** | -$9,302,710 | -10.0 | Neural network stability |
| **Max Reward** | +$10,000,000 | +10.0 | Neural network stability |
| **Drawdown Scale** | λ_dd = 5,688,000 | Normalized quadratic | No gradient explosion |
| **SAC Loss** | MSE on 10⁷² | MSE on 10² | Stable gradients |

## Key Insight

**Neural networks (and SAC) require normalized inputs**. Training on raw dollar values ($19,180) is mathematically equivalent to training on values of ~10⁷ in float32, which causes:

1. **Gradient overflow**: Squared values exceed float32 range
2. **Numerical instability**: Division by near-zero variances
3. **NaN propagation**: Infects entire network

**The solution**: Always normalize financial values to [-10, +10] range before feeding to neural networks.
