"""
Configuration Management Module
Loads and validates trading configuration from YAML and environment variables.
"""
import os
import yaml
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

# Try to load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class BrokerConfig:
    name: str = "alpaca"
    paper_trading: bool = True
    api_base_url: str = "https://paper-api.alpaca.markets"
    api_data_url: str = "https://data.alpaca.markets"
    websocket_url: str = "wss://stream.data.alpaca.markets/v2/sip"
    api_key: str = field(default_factory=lambda: os.getenv("ALPACA_API_KEY", ""))
    secret_key: str = field(default_factory=lambda: os.getenv("ALPACA_SECRET", ""))


@dataclass
class TimezoneConfig:
    local_tz: str = "Europe/Madrid"
    market_tz: str = "America/New_York"
    scan_start: str = "09:45"
    entry_window_start: str = "09:45"
    entry_window_end: str = "14:30"
    flatten_time: str = "15:25"
    market_open: str = "09:30"
    market_close: str = "16:00"


@dataclass
class ScreeningConfig:
    min_percent_gain: float = 60.0
    max_percent_gain: float = 500.0
    min_price: float = 2.0
    max_price: float = 50.0
    min_volume: int = 500000
    max_float_millions: int = 100


@dataclass
class VolumeExhaustionConfig:
    peak_lookback_minutes: int = 390
    entry_threshold: float = 0.60
    add2_threshold: float = 0.50
    add3_threshold: float = 0.40
    price_proximity_to_high: float = 0.95
    new_high_required_for_add: bool = True


@dataclass
class SignalsConfig:
    vwap_extension_threshold: float = 1.20
    vwap_anchor: str = "session"
    min_exhaustion_factors: int = 2
    absorption_lookback_ticks: int = 50
    momentum_divergence_periods: int = 3
    min_minutes_between_adds: int = 10


@dataclass
class ScalingConfig:
    enabled: bool = True
    initial_size_percent: int = 25
    add2_size_percent: int = 25
    add3_size_percent: int = 50
    max_adds: int = 3
    max_shares_per_position: int = 5000
    max_position_value: float = 30000.0


@dataclass
class LeverageConfig:
    enabled: bool = False
    max_leverage: float = 1.0
    margin_buffer: float = 0.15


@dataclass
class ExitsConfig:
    tp1_enabled: bool = True
    tp1_percent: int = 35
    tp1_target: str = "vwap"
    tp2_enabled: bool = True
    tp2_percent: int = 35
    tp2_percent_drop: float = 8.0
    tp3_enabled: bool = True
    tp3_percent: int = 30
    tp3_percent_drop: float = 15.0
    use_trailing_after_tp1: bool = True
    trailing_activation_percent: float = 3.0


@dataclass
class RiskConfig:
    max_portfolio_risk_percent: float = 1.0
    initial_stop_percent: float = 4.0
    average_stop_percent: float = 3.5
    daily_loss_limit_percent: float = 2.0
    max_positions: int = 3
    max_daily_trades: int = 9
    atr_lookback: int = 14
    atr_multiplier_stop: float = 2.0
    min_account_equity: int = 2000
    maintenance_margin_percent: float = 100.0


@dataclass
class ExecutionConfig:
    order_type: str = "limit"
    limit_offset_ticks: int = 2
    time_in_force: str = "ioc"
    hard_stop_at_apex: bool = True
    flatten_before_close_minutes: int = 35


@dataclass
class ComplianceConfig:
    spain_homogeneous_loss_rule: bool = False
    blacklist_duration_days: int = 365
    no_overnight_positions: bool = True
    flat_before_close_minutes: int = 35


@dataclass
class DataConfig:
    tick_buffer_size: int = 10000
    bar_aggregation_seconds: int = 60
    volume_lookback_minutes: int = 5
    use_lazyframe: bool = True


@dataclass
class PerformanceConfig:
    numba_cache: bool = True
    numba_fastmath: bool = True
    numba_parallel: bool = True
    polars_threads: int = 0
    websocket_auto_reconnect: bool = True
    reconnect_delay_seconds: int = 5


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "logs/trading_engine.log"
    max_size_mb: int = 100
    backup_count: int = 10
    format: str = "json"


@dataclass
class Config:
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    timezone: TimezoneConfig = field(default_factory=TimezoneConfig)
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    volume_exhaustion: VolumeExhaustionConfig = field(default_factory=VolumeExhaustionConfig)
    signals: SignalsConfig = field(default_factory=SignalsConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    leverage: LeverageConfig = field(default_factory=LeverageConfig)
    exits: ExitsConfig = field(default_factory=ExitsConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)
    data: DataConfig = field(default_factory=DataConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(config_path: str = "config/settings.yaml") -> Config:
    """Load configuration from YAML file."""
    path = Path(config_path)
    
    if not path.exists():
        return Config()
    
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    return Config(
        broker=BrokerConfig(**data.get('broker', {})),
        timezone=TimezoneConfig(**data.get('timezone', {})),
        screening=ScreeningConfig(**data.get('screening', {})),
        volume_exhaustion=VolumeExhaustionConfig(**data.get('volume_exhaustion', {})),
        signals=SignalsConfig(**data.get('signals', {})),
        scaling=ScalingConfig(**data.get('scaling', {})),
        leverage=LeverageConfig(**data.get('leverage', {})),
        exits=ExitsConfig(**data.get('exits', {})),
        risk=RiskConfig(**data.get('risk', {})),
        execution=ExecutionConfig(**data.get('execution', {})),
        compliance=ComplianceConfig(**data.get('compliance', {})),
        data=DataConfig(**data.get('data', {})),
        performance=PerformanceConfig(**data.get('performance', {})),
        logging=LoggingConfig(**data.get('logging', {}))
    )


# Global config instance
CONFIG = load_config()
