"""
Rule-Based Baseline Agent - Deterministic V5 Relaxed Strategy

This agent implements the deterministic trading rules from the V5 Relaxed
strategy without any learning. It serves as the performance floor that
RL must beat to justify its complexity.

Strategy Rules:
1. Entry: Short when VWAP deviation > threshold AND in entry window
2. Exit: Cover when price touches VWAP OR stop loss hit
3. Position sizing: 
   - fixed_shares: Fixed share count
   - fixed_fraction_of_equity: Fixed fraction of current equity (matches RL convention)
4. No overnight positions: Flat by end of day

This agent uses ZERO learned parameters and ZERO state encoding.
If RL cannot beat this, the RL approach is not viable.
"""

import numpy as np
from typing import Dict, Optional, Any, Literal
from dataclasses import dataclass


@dataclass
class RuleAgentConfig:
    """Configuration for rule-based agent."""
    # Entry criteria
    entry_vwap_threshold: float = 20.0  # VWAP deviation % to enter short (matches settings.yaml)
    entry_time_start: tuple = (9, 45)   # (hour, minute) ET
    entry_time_end: tuple = (14, 30)    # (hour, minute) ET
    
    # Exit criteria
    stop_loss_pct: float = 4.0          # Stop loss from entry price
    take_profit_target: str = "vwap"    # Cover at VWAP
    flatten_time: tuple = (15, 25)      # Flatten by this time
    
    # Position sizing mode
    sizing_mode: Literal["fixed_shares", "fixed_fraction_of_equity"] = "fixed_shares"
    
    # For fixed_shares mode
    position_size_shares: int = 100     # Fixed shares per trade
    
    # For fixed_fraction_of_equity mode (matches RL convention)
    target_exposure_fraction: float = 0.30  # Target 30% of equity short
    max_position_value: float = 30000.0     # Hard cap at $30k (matches RL)
    
    # Risk limits
    max_trades_per_episode: int = 5     # Limit churn


class RuleBasedAgent:
    """
    Deterministic rule-based trading agent with configurable sizing.
    
    This agent implements the V5 Relaxed strategy with fixed rules.
    No learning, no state encoding, no neural networks.
    
    Two sizing modes:
    1. fixed_shares: Simple fixed share count (original behavior)
    2. fixed_fraction_of_equity: Size as fraction of capital (matches RL convention)
    
    Usage:
        agent = RuleBasedAgent()
        action = agent.act(observation, info)
    """
    
    def __init__(self, config: Optional[RuleAgentConfig] = None):
        self.config = config or RuleAgentConfig()
        self.in_position = False
        self.entry_price = 0.0
        self.position_size = 0
        self.trades_this_episode = 0
        
    def reset(self):
        """Reset agent state for new episode."""
        self.in_position = False
        self.entry_price = 0.0
        self.position_size = 0
        self.trades_this_episode = 0
        
    def act(self, observation: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        """
        Deterministic action selection based on rules.
        
        Args:
            observation: Environment observation (ignored—rules use info)
            info: Dict with 'vwap_deviation', 'price', 'vwap', 'time', 
                  'position', 'capital', etc.
            
        Returns:
            action: np.ndarray with continuous action in [-1, 0]
                   -1 = full short, 0 = flat, values in between = partial
                   
        Note:
            The environment interprets action as target exposure fraction.
            For fixed_shares mode, we return -1 (full short) and let env handle sizing.
            For fixed_fraction_of_equity mode, we return the target fraction.
        """
        # Extract state from info
        vwap_dev = info.get('vwap_deviation', 0.0)
        price = info.get('price', 0.0)
        vwap = info.get('vwap', price)
        current_time = info.get('time', None)
        current_position = info.get('position', 0.0)
        current_capital = info.get('capital', 100000.0)
        
        # Update internal state tracking
        self.in_position = current_position < 0  # Negative = short
        
        # Check time constraints
        in_entry_window = self._is_in_entry_window(current_time)
        must_flatten = self._is_flatten_time(current_time)
        
        # Must flatten by end of day
        if must_flatten and self.in_position:
            return np.array([0.0])  # Go flat
        
        # If in position, check exit conditions
        if self.in_position:
            # Stop loss check
            if price > self.entry_price * (1 + self.config.stop_loss_pct / 100):
                return np.array([0.0])  # Stop out
            
            # Take profit at VWAP
            if self.config.take_profit_target == "vwap" and price <= vwap:
                return np.array([0.0])  # Cover at VWAP
            
            # Hold position
            return None  # No action (hold)
        
        # Not in position—check entry conditions
        if not in_entry_window:
            return None  # No entry outside window
        
        if self.trades_this_episode >= self.config.max_trades_per_episode:
            return None  # Max trades reached
        
        # Entry: VWAP deviation above threshold
        if vwap_dev > self.config.entry_vwap_threshold:
            self.entry_price = price
            self.trades_this_episode += 1
            
            # Return action based on sizing mode
            if self.config.sizing_mode == "fixed_shares":
                # Return full short signal - env will handle fixed shares
                return np.array([-1.0])
            elif self.config.sizing_mode == "fixed_fraction_of_equity":
                # Calculate target position value as fraction of capital
                target_value = current_capital * self.config.target_exposure_fraction
                # Clamp to max position value
                target_value = min(target_value, self.config.max_position_value)
                # Convert to action fraction (negative for short)
                # Max short action is -1.0, so scale accordingly
                # For simplicity: just return the target fraction capped at -1.0
                action = -min(target_value / current_capital, 1.0)
                return np.array([action])
            else:
                raise ValueError(f"Unknown sizing mode: {self.config.sizing_mode}")
        
        # No action
        return None
    
    def _is_in_entry_window(self, current_time) -> bool:
        """Check if current time is within entry window."""
        if current_time is None:
            return False
        start_h, start_m = self.config.entry_time_start
        end_h, end_m = self.config.entry_time_end
        
        start_minutes = start_h * 60 + start_m
        end_minutes = end_h * 60 + end_m
        current_minutes = current_time.hour * 60 + current_time.minute
        
        return start_minutes <= current_minutes <= end_minutes
    
    def _is_flatten_time(self, current_time) -> bool:
        """Check if we must flatten position."""
        if current_time is None:
            return False
        flatten_h, flatten_m = self.config.flatten_time
        current_minutes = current_time.hour * 60 + current_time.minute
        flatten_minutes = flatten_h * 60 + flatten_m
        return current_minutes >= flatten_minutes


def create_v5_relaxed_agent_fixed_shares(shares: int = 100) -> RuleBasedAgent:
    """Factory function for V5 Relaxed rule agent with fixed shares."""
    config = RuleAgentConfig(
        sizing_mode="fixed_shares",
        entry_vwap_threshold=20.0,
        stop_loss_pct=4.0,
        position_size_shares=shares
    )
    return RuleBasedAgent(config)


def create_v5_relaxed_agent_fraction_of_equity(
    fraction: float = 0.30,
    max_position: float = 30000.0
) -> RuleBasedAgent:
    """
    Factory function for V5 Relaxed rule agent with equity-fraction sizing.

    This matches the RL convention of sizing as fraction of capital
    with a hard maximum position value cap.

    Args:
        fraction: Target exposure as fraction of equity (e.g., 0.30 = 30%)
        max_position: Hard cap in dollars (matches RL's max_position_value)
    """
    config = RuleAgentConfig(
        sizing_mode="fixed_fraction_of_equity",
        entry_vwap_threshold=20.0,
        stop_loss_pct=4.0,
        target_exposure_fraction=fraction,
        max_position_value=max_position
    )
    return RuleBasedAgent(config)


def run_baseline(
    entry_price: float,
    exit_price: float,
    shares: int,
    side: str = "short",
    transaction_cost_bps: float = 30.0,
) -> float:
    """
    Compute net PnL for a single rule-baseline round-trip trade,
    charging a per-leg transaction cost to match ``src/rl/env.py``.

    The RL environment charges 30bps on each leg of every trade. For RL-vs-rule
    comparisons to be honest, the rule baseline must charge the same. This
    function is the public entry-point that callers (e.g., the comparison
    harness or future analytics) use to compute a trade's net PnL with costs.

    Cost model (per leg):
        leg_fee = leg_price * shares * (transaction_cost_bps / 10_000.0)

    Net PnL:
        short: net = (entry_price - exit_price) * shares - entry_fee - exit_fee
        long:  net = (exit_price - entry_price) * shares - entry_fee - exit_fee

    Args:
        entry_price: Fill price on the opening leg.
        exit_price:  Fill price on the closing leg.
        shares:      Share count (>= 0).
        side:        ``"short"`` (default — matches the parabolic strategy) or ``"long"``.
        transaction_cost_bps: Per-leg cost in basis points. Default 30.0 matches
            ``src/rl/env.py``'s ``transaction_cost_bps`` so RL-vs-rule comparison
            stops being noise.

    Returns:
        Net realized PnL in dollars, after charging the cost on BOTH legs.
    """
    if shares < 0:
        raise ValueError(f"shares must be >= 0, got {shares}")
    if transaction_cost_bps < 0:
        raise ValueError(f"transaction_cost_bps must be >= 0, got {transaction_cost_bps}")

    side_norm = side.lower().strip()
    if side_norm == "short":
        gross_pnl = (entry_price - exit_price) * shares
    elif side_norm == "long":
        gross_pnl = (exit_price - entry_price) * shares
    else:
        raise ValueError(f"side must be 'short' or 'long', got {side!r}")

    # Per-leg cost: entry_price * shares * (bps/10000.0) on entry, same on exit.
    cost_fraction = transaction_cost_bps / 10_000.0
    entry_fee = entry_price * shares * cost_fraction
    exit_fee = exit_price * shares * cost_fraction

    return gross_pnl - entry_fee - exit_fee
