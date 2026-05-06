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
