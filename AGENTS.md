# Parabolic Reversal Trading Engine - Agent Guide

## Project Overview

This is a **high-performance algorithmic trading system** designed to fade parabolic price reversals in micro-cap equities. The system implements the "First Red Day" and "Blow-Off Top" short-selling strategies targeting stocks with 60-500% intraday gains showing exhaustion patterns.

**Core Strategy**: Short overextended stocks when price extends >120% above VWAP with volume exhaustion, covering at VWAP mean reversion or stopping at parabolic apex.

**Key Differentiator**: Unlike traditional multi-day "First Red Day" strategies, this system focuses on **intraday** parabolic moves only - no requirement for consecutive up days.

**Version**: 2.0 (Progressive Exhaustion Scale-In Strategy)

**Primary Language**: English (all code comments and documentation)

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Broker | Alpaca Markets API | Trade execution & market data |
| Data Processing | Polars 0.20+ | High-performance DataFrames (30-50x faster than Pandas) |
| Numerical Computing | Numba 0.58+ | JIT compilation for C-level execution |
| Memory Format | PyArrow 14+ | Zero-copy memory transfers |
| Streaming | WebSocket (IEX/SIP feed) | Real-time tick data |
| Language | Python 3.9+ | Primary development language |
| Logging | structlog + python-json-logger | Structured JSON logging |
| Data Storage | Parquet | Historical data cache format |

---

## Project Structure

```
quant_trading/
├── config/
│   └── settings.yaml              # Strategy configuration parameters
├── src/
│   ├── main_engine.py             # Live trading orchestrator (entry point)
│   ├── data/
│   │   ├── alpaca_client.py       # WebSocket + REST API client
│   │   └── polars_engine.py       # High-performance data processing
│   ├── indicators/
│   │   └── numba_kernels.py       # JIT-compiled VWAP/ATR calculations
│   ├── screening/
│   │   └── screener.py            # Asset qualification & screening
│   ├── risk/
│   │   ├── position_manager.py    # Risk management & position sizing
│   │   ├── ml_risk_manager.py     # ML-based risk assessment
│   │   └── ml/                    # ML components
│   │       ├── bayesian_inference.py
│   │       ├── ensemble_models.py
│   │       ├── feature_engineering.py
│   │       ├── model_validator.py
│   │       ├── online_learning.py
│   │       └── risk_metrics.py
│   ├── execution/
│   │   └── signal_engine.py       # Entry/exit signal generation
│   ├── backtest/
│   │   ├── backtest_engine.py     # Bar-based backtesting
│   │   ├── tick_backtest_engine.py    # Tick-level backtesting
│   │   ├── tick_backtest_engine_v2.py # V2 improvements
│   │   ├── tick_backtest_engine_v3.py # V3 improvements
│   │   ├── tick_backtest_engine_v4.py # V4 improvements
│   │   ├── tick_backtest_engine_v5.py # V5 strict entry
│   │   ├── tick_backtest_engine_v6.py # V6 relaxed entry
│   │   ├── tick_backtest_engine_v7.py # V7 improvements
│   │   ├── tick_backtest_engine_v8.py # V8 improvements
│   │   ├── tick_backtest_engine_v9.py # V9 improvements
│   │   ├── tick_backtest_engine_v10.py # V10 fast exit
│   │   ├── historical_screener.py # Historical setup scanner
│   │   ├── historical_tick_fetcher.py # Alpaca tick data fetcher
│   │   ├── batch_backtest.py      # Multi-year batch testing
│   │   ├── extended_universe.py   # 3,527 micro-cap symbols
│   │   ├── data_fetcher.py        # Data fetching utilities
│   │   ├── visualizer.py          # HTML/Chart reports
│   │   └── trade_visualizer.py    # Trade-specific visualization
│   ├── strategies/
│   │   ├── strategy_registry.py   # Strategy registration
│   │   ├── v5_strict.py           # Strict entry criteria
│   │   ├── v5_relaxed_scanner.py  # Relaxed discovery (WINNER)
│   │   ├── v5_institutional.py    # Institutional version
│   │   ├── v5_ml_risk.py          # ML-enhanced risk
│   │   └── v6_relaxed_entry.py    # V6 entry logic
│   └── utils/
│       ├── config.py              # Configuration management (dataclasses)
│       └── logger.py              # Structured logging
├── data/cache/                    # Historical data cache (Parquet files)
├── logs/                          # Trading logs (JSON format)
├── reports/                       # Generated backtest reports
├── run.py                         # Live trading launcher
├── run_historical_backtest.py     # Historical backtest runner
├── run_complete_fresh_backtest.py # Full fresh backtest
├── run_full_3571_backtest.py      # Full symbol universe backtest
├── scan_extended_universe.py      # Scan 3,527 symbols for setups
├── show_setups.py                 # Display found setups
├── test_engine.py                 # Component validation tests
├── test_tick_backtest.py          # Backtest validation tests
├── test_connection.py             # API connectivity tests
├── test_ml_risk.py                # ML risk system tests
├── requirements.txt               # Python dependencies
└── .env                           # API credentials (gitignored)
```

---

## Build and Run Commands

### Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. Create `.env` file in project root:
```bash
ALPACA_API_KEY=your_key_here
ALPACA_SECRET=your_secret_here
```

2. Edit `config/settings.yaml` for strategy parameters (see Key Configuration Parameters section)

### Running the System

**Live Trading (Paper):**
```bash
python run.py
```

**Historical Backtesting:**
```bash
# Quick test (10 setups)
python run_historical_backtest.py --quick-test

# Full backtest (all setups, 30+ minutes)
python run_historical_backtest.py --full

# Test specific setup
python run_historical_backtest.py --symbol AMC --date 2021-06-02

# Scan only (no backtest)
python run_historical_backtest.py --scan
```

**Complete Fresh Backtest (Full 6-Year):**
```bash
# Using batch file (recommended)
START_COMPLETE_BACKTEST.bat

# Or directly
python run_complete_fresh_backtest.py
```

**Full 3,571 Symbol Backtest:**
```bash
# Using batch file
run_full_backtest.bat

# Or directly
python run_full_3571_backtest.py
```

**Extended Universe Scanning:**
```bash
# Scan 3,527 micro-cap symbols
python scan_extended_universe.py

# Show found setups
python show_setups.py
```

**Testing:**
```bash
# Test all engine components
python test_engine.py

# Test tick-level backtest
python test_tick_backtest.py

# Test API connection
python test_connection.py

# Test ML risk system
python test_ml_risk.py
```

---

## Key Configuration Parameters

Located in `config/settings.yaml`:

```yaml
# Screening Criteria (Intraday Only)
screening:
  min_percent_gain: 60.0           # Minimum % gain from open
  max_percent_gain: 500.0          # Avoid extreme outliers
  min_price: 2.0                   # Avoid sub-$2 stocks
  max_price: 50.0                  # Focus on micro/small cap
  min_volume: 500000               # Minimum daily volume
  max_float_millions: 100          # Low float preference

# Volume Exhaustion Detection
volume_exhaustion:
  peak_lookback_minutes: 390       # Full session lookback
  entry_threshold: 0.60            # Volume < 60% of peak = exhaustion
  add2_threshold: 0.50             # Volume < 50% = strong exhaustion
  add3_threshold: 0.40             # Volume < 40% = severe exhaustion
  price_proximity_to_high: 0.95    # Within 5% of HOD
  new_high_required_for_add: true  # Must make new high on lower volume

# Signal Detection
signals:
  vwap_extension_threshold: 1.20   # Price > 120% of VWAP
  vwap_anchor: "session"           # VWAP from 9:30 AM
  min_exhaustion_factors: 2        # Minimum confirming factors
  absorption_lookback_ticks: 50
  momentum_divergence_periods: 3
  min_minutes_between_adds: 10     # Cooldown between adds

# Progressive Position Building (Scale-In)
scaling:
  enabled: true
  initial_size_percent: 25         # First entry: 25%
  add2_size_percent: 25            # Second: 25% (50% total)
  add3_size_percent: 50            # Third: 50% (100% total)
  max_adds: 3
  max_shares_per_position: 5000
  max_position_value: 30000

# Risk Management
risk:
  max_portfolio_risk_percent: 1.0  # 1% max risk per position
  initial_stop_percent: 4.0        # 4% stop on initial entry
  average_stop_percent: 3.5        # 3.5% stop on full position
  daily_loss_limit_percent: 2.0    # Stop after -2% account loss
  max_positions: 3                 # Max concurrent positions
  max_daily_trades: 9              # 3 positions x 3 adds
  atr_lookback: 14
  atr_multiplier_stop: 2.0

# Exit Targets (Layered Profit Taking)
exits:
  tp1_enabled: true
  tp1_percent: 35                  # Close 35% at VWAP
  tp1_target: "vwap"
  tp2_enabled: true
  tp2_percent: 35                  # Close 35% at -8% from entry
  tp2_percent_drop: 8.0
  tp3_enabled: true
  tp3_percent: 30                  # Close 30% at -15% from entry
  tp3_percent_drop: 15.0

# Trading Schedule
timezone:
  local_tz: "Europe/Madrid"
  market_tz: "America/New_York"
  scan_start: "09:45"
  entry_window_start: "09:45"
  entry_window_end: "14:30"
  flatten_time: "15:25"
```

---

## Code Style Guidelines

### Python Conventions
- **Type hints**: Use `typing` module for all function signatures
- **Docstrings**: Google-style docstrings for all public functions
- **Imports**: Group as: stdlib, third-party, local (each group alphabetically)
- **Line length**: 100 characters maximum
- **Naming**:
  - `snake_case` for functions/variables
  - `PascalCase` for classes
  - `UPPER_CASE` for constants
  - `_prefix` for private methods

### Performance-Critical Code
- Use Numba `@njit` decorator for numerical calculations
- Convert Polars/NumPy to NumPy arrays before Numba calls
- Use Polars LazyFrames for complex queries
- Avoid pandas - use Polars exclusively

### Example Pattern:
```python
from typing import Dict, Optional
import numpy as np
from numba import njit
import polars as pl

from src.utils.config import CONFIG
from src.utils.logger import logger


def calculate_metrics(data: pl.DataFrame) -> Dict[str, float]:
    """
    Calculate trading metrics from bar data.
    
    Args:
        data: Polars DataFrame with OHLCV columns
        
    Returns:
        Dictionary of calculated metrics
    """
    # Convert to numpy for Numba processing
    highs = data['high'].to_numpy()
    lows = data['low'].to_numpy()
    
    # Use Numba kernel for performance
    result = _numba_calculation(highs, lows)
    
    return {'metric': result}


@njit(cache=True, fastmath=True)
def _numba_calculation(highs: np.ndarray, lows: np.ndarray) -> float:
    """Internal Numba-compiled calculation."""
    return np.mean(highs - lows)
```

---

## Testing Instructions

### Unit Testing
No formal test suite exists (no pytest configuration). Test via:
1. Run `test_engine.py` for component validation
2. Run `test_tick_backtest.py` for backtest validation
3. Run `test_connection.py` for API connectivity
4. Run `test_ml_risk.py` for ML risk system

### Backtest Validation
Before any code changes affecting trading logic:
```bash
# Run quick test to ensure nothing broke
python run_historical_backtest.py --quick-test

# Compare results with previous runs
# Reports saved to reports/batch_backtest_*.csv
```

### Live Trading Precautions
- Always test in paper trading first (`paper_trading: true` in config)
- Verify WebSocket connection before market open
- Check account equity meets $2,000 minimum for short selling

---

## Security Considerations

### API Credentials
- Store in `.env` file (already in `.gitignore`)
- Never commit credentials to version control
- Use paper trading API keys for development
- Rotate keys every 90 days

### Trading Safeguards
- Hard stops at parabolic apex (prevents unlimited loss)
- Position sizing caps at 1% portfolio risk per trade
- Auto-flatten 15 minutes before market close
- Max 3 concurrent positions limit
- Spain Homogeneous Loss Rule compliance (365-day blacklist)

### Data Handling
- Historical data cached locally in `data/cache/`
- Cache files are Parquet format (efficient storage)
- No sensitive data logged (only trade P&L, not account details)

---

## Architecture Patterns

### Data Flow (Live Trading)
1. **WebSocket** (`alpaca_client.py`) receives tick data
2. **Polars Engine** (`polars_engine.py`) aggregates to bars
3. **Signal Engine** (`signal_engine.py`) evaluates entry/exit
4. **Risk Manager** (`position_manager.py`) sizes positions
5. **Alpaca Client** executes orders via REST API

### Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `TradingEngine` | `main_engine.py` | Main orchestrator |
| `AlpacaClient` | `alpaca_client.py` | Broker interface |
| `PolarsSignalEngine` | `polars_engine.py` | Data processing |
| `ParabolicScreener` | `screener.py` | Asset qualification |
| `RiskManager` | `position_manager.py` | Risk controls |
| `ParabolicSignalEngine` | `signal_engine.py` | Signal generation |
| `MLRiskManager` | `ml_risk_manager.py` | ML-based risk |

### Numba Kernels
Located in `src/indicators/numba_kernels.py`:
- `calculate_vwap_numba()` - Volume-weighted average price
- `calculate_atr_numba()` - Average true range (volatility)
- `calculate_position_size_numba()` - Volatility-based sizing
- `detect_absorption_numba()` - Iceberg order detection
- `detect_momentum_divergence_numba()` - Price-volume divergence

---

## Strategy Versions

### V5 Relaxed Scanner (RECOMMENDED)
**Performance (3,527 symbols, 2019-2024):**
- Setups Found: 909 (vs 242 original)
- Trades Taken: 327 (vs 40 original)
- Win Rate: 78.9% (maintained)
- Total P&L: +$580,381 (11x improvement)
- Trades/Year: ~54

**Key Insight**: Relax DISCOVERY to find more setups, keep ENTRY strict to maintain win rate.

### V10 Fast Exit
Latest backtest engine with:
- Stop checked against HIGH price (not CLOSE)
- Tighter 2% stop
- 30-minute time stop
- Faster exits in parabolic squeezes

---

## Debugging and Troubleshooting

### Common Issues

**No trades executed:**
- Check VWAP extension threshold in config
- Verify execution window times (ET timezone)
- Ensure date had parabolic movement (>60% gain)

**WebSocket disconnects:**
- Auto-reconnect is enabled by default
- Check API credentials
- Verify internet connection

**Slow backtesting:**
- First run downloads data (subsequent runs use cache)
- Use `--quick-test` for faster validation
- Check disk space for cache files

### Debug Scripts
```bash
# Check timezone handling
python debug_timezone.py

# Diagnose backtest issues
python diagnose_backtest.py

# Check VWAP calculations
python check_vwap.py

# Check signal generation
python check_signals.py
```

### Logging
- Logs stored in `logs/trading_engine.log`
- JSON format for structured parsing
- Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Rotate at 100MB, keep 10 backups

---

## External Dependencies

### Alpaca Markets API
- **Docs**: https://alpaca.markets/docs/
- **Rate Limits**: 200 requests/minute (free tier)
- **Data Retention**: 6+ years historical
- **WebSocket**: IEX (free) or SIP (paid) feed

### Python Packages (Key)
```
polars>=0.20.0      # DataFrame processing
numba>=0.58.0       # JIT compilation
alpaca-py>=0.15.0   # Broker API
websockets>=12.0    # WebSocket client
pyyaml>=6.0.1       # Config parsing
pydantic>=2.5.0     # Data validation
python-dotenv>=1.0.0 # Environment variables
structlog>=23.2.0   # Structured logging
```

---

## Compliance and Regulations

### Spain Homogeneous Loss Rule
- Enforced when `spain_homogeneous_loss_rule: true` in config
- Blacklists symbols for 365 days after realized loss
- Prevents tax liability from wash sale-like activity

### Short Selling Requirements
- Minimum account equity: $2,000
- Easy-to-Borrow (ETB) stocks only
- No overnight positions (flatten by 15:25 ET)
- Maintenance margin: 100% for sub-$5 stocks

---

## Performance Benchmarks

| Metric | Target | Actual |
|--------|--------|--------|
| Tick-to-signal latency | < 10ms | ~5ms |
| VWAP calculation | < 1ms | ~0.5ms |
| Order submission | < 50ms | ~30ms |
| Memory usage | < 500MB | ~300MB |

---

## File Ownership and Modifications

When modifying files, maintain:
1. Existing code style and patterns
2. Type hints on all new functions
3. Docstrings for public APIs
4. Numba decorators for numerical code
5. Structured logging (no print statements)

### Critical Files (Modify with Care)
- `src/main_engine.py` - Core trading logic
- `src/risk/position_manager.py` - Risk controls
- `src/execution/signal_engine.py` - Entry/exit logic
- `config/settings.yaml` - Strategy parameters

---

## Quick Reference

**Start Live Trading:**
```bash
python run.py
```

**Run Backtest:**
```bash
python run_historical_backtest.py --quick-test
```

**Full Fresh Backtest:**
```bash
START_COMPLETE_BACKTEST.bat
```

**Scan for Setups:**
```bash
python scan_extended_universe.py
python show_setups.py
```

**View Logs:**
```powershell
Get-Content logs/trading_engine.log -Tail 50
```

**Check Config:**
```bash
type config\settings.yaml
```

---

## Important Notes for AI Agents

1. **No pyproject.toml or setup.py**: This project uses simple pip requirements, not a package structure
2. **No pytest**: Tests are standalone scripts, not using pytest framework
3. **Data caching**: All historical data is cached in `data/cache/` as Parquet files - DO NOT delete these
4. **Timezone handling**: System operates in America/New_York (ET) but can be configured for other timezones
5. **ML components**: ML risk system is optional but recommended - thresholds stored in `src/risk/ml/`
6. **Backtest engines**: Multiple versions exist (V1-V10) - V5 relaxed scanner is the recommended strategy
7. **Position sizing**: Uses progressive scale-in (3 adds) with decreasing risk per add

---

*Last Updated: 2026-03-12*
*Project Status: Production-ready for paper trading*
*Recommended Strategy: V5 Relaxed Scanner*
