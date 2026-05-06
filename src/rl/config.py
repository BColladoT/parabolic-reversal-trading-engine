"""RL Configuration - shared settings."""

# Default configuration matching EnvironmentConfig defaults
RL_CONFIG = {
    'min_vwap_deviation_entry': 15.0,  # VWAP threshold for entry (lowered from 20% to capture more setups)
    'max_single_trade_loss': -15000.0,
    'max_drawdown': -15000.0,
    'circuit_breaker_threshold': -15000.0,
    'kelly_fraction': 0.25,
    'max_leverage_cap': 3.0,
    'min_leverage_floor': 0.5,
    'max_shares_per_position': 5000,
    'max_position_value': 30000.0,
    # Layered position safety (added after -$88K catastrophic loss in full WFO)
    'intra_step_stop_loss': -2000.0,        # Force-close if unrealized PnL < this after bar advance
    'max_position_capital_fraction': 0.30,   # Position ≤ 30% of current capital
}
