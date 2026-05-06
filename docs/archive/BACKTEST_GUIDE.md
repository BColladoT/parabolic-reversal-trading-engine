# Parabolic Reversal Strategy - Backtesting Guide

## Overview

We've built a **tick-level backtesting system** using Alpaca's historical trade data API. This gives you ultra-accurate simulation with full audit trails showing exactly why each trade was made.

## What You Can Now Do

### 1. Fetch Historical Tick Data

```python
from src.backtest.historical_tick_fetcher import tick_fetcher
from datetime import datetime

# Fetch actual executed trades
trades = tick_fetcher.fetch_historical_trades(
    symbol="TSLA",
    start=datetime(2025, 3, 3, 10, 0),  # 10:00 AM
    end=datetime(2025, 3, 3, 11, 0),    # 11:00 AM
    use_cache=True
)

# Each trade includes:
# - timestamp (nanosecond precision)
# - price (actual execution price)
# - size (number of shares)
# - exchange (V = IEX, etc.)
# - conditions (trade conditions)
```

### 2. Run Tick-Level Backtest

```bash
python test_tick_backtest.py
```

Output shows:
```
Sample tick data (first 5 trades):
Time            Price      Size       Exchange  
--------------------------------------------------
15:00:02.376    $292.98    15.0       V         
15:00:02.684    $292.70    10.0       V         
15:00:02.686    $292.70    6.0        V         
15:00:04.925    $292.77    10.0       V         
15:00:05.182    $292.65    100.0      V         
```

### 3. View Detailed Trade Reasoning

Every entry shows:
```
[ENTRY] 10:15:32.450 TSLA @ $310.25 (slippage: +5bps)
  Reasoning: VWAP extension 1.25x (>115%); Volume exhaustion detected; ATR $2.50
  Confidence: 75% | Criteria met: 3/4
  Size: 150 shares | Risk: $375.00
  Stop: $315.25 | Target: $302.50 (VWAP)
```

Every exit shows:
```
[EXIT] 10:32:15.120 @ $305.50 (slippage: -5bps)
  P&L: +$712.50 (+1.53%) | Reason: profit_target
  Hold time: 0:16:42.670
```

### 4. Generate HTML Reports

Reports include:
- **Performance metrics** (win rate, profit factor, drawdown)
- **Trade log** with full reasoning
- **Detailed audit** showing market conditions at entry
- **Charts** (equity curve, trade markers)

Open `reports/backtest_[SYMBOL]_[DATE].html` in your browser.

## Data Sources

### Alpaca Historical API (Free Tier)

| Data Type | Endpoint | Granularity | Retention |
|-----------|----------|-------------|-----------|
| **Trades** | `/v2/stocks/{symbol}/trades` | Tick-level | 6+ years |
| **Quotes** | `/v2/stocks/{symbol}/quotes` | Tick-level | 6+ years |
| **Bars** | `/v2/stocks/{symbol}/bars` | 1Min/5Min/1D | 6+ years |

### What's Available

**Historical Trades:**
- Actual executed transaction prices
- Trade sizes (share quantities)
- Exchange where trade occurred
- Trade conditions (@ = regular, F = intermarket sweep, etc.)

**Historical Quotes:**
- Bid/ask prices and sizes
- Quote timestamps
- Best bid/offer (BBO)

## Backtest Accuracy

### What We Simulate

| Feature | Simulation Method |
|---------|-------------------|
| **Fill Prices** | Actual trade prices + slippage (5bps) |
| **Market Impact** | Volume-weighted average of available liquidity |
| **Slippage** | Configurable (default: 0.05% on entry/exit) |
| **Timing** | Tick-by-tick evaluation (nanosecond precision) |
| **VWAP** | Calculated from all trades, not just bars |
| **ATR** | Real-time from tick-aggregated bars |

### Entry Criteria Evaluated

1. **VWAP Extension** > 115% (price vs VWAP)
2. **Volume Exhaustion** < 60% of average
3. **Price Range** > 2% intraday
4. **Trend Confirmation** Price above VWAP

Requires **3+ criteria met** with **60%+ confidence**

### Exit Conditions

- **Stop Loss**: ATR-based or parabolic apex
- **Profit Target**: VWAP mean reversion
- **Time-Based**: End of 10:00-11:00 execution window

## Example: Analyzing a Parabolic Move

```python
from src.backtest.tick_backtest_engine import tick_backtest_engine
from datetime import datetime

# Find a day with major gain (e.g., TSLA +80% day)
result = tick_backtest_engine.run_tick_backtest(
    symbol="TSLA",
    date=datetime(2024, 1, 15),  # Use actual volatile date
    verbose=True
)

# Results include:
# - Total trades executed
# - Win rate
# - Total P&L
# - Average trade duration
# - Detailed audit trail
```

## Finding Volatile Dates

To find dates with parabolic moves:

```python
from src.backtest.data_fetcher import data_fetcher

# Scan for 80%+ gainers in last 30 days
candidates = data_fetcher.find_parabolic_candidates(
    symbols=['TSLA', 'AAPL', 'NVDA', 'AMD'],
    lookback_days=30,
    min_gain_percent=80.0
)

for c in candidates:
    print(f"{c['symbol']} | {c['date']} | +{c['gain_percent']:.1f}%")
```

## Caching System

All fetched data is cached locally:
```
data/cache/ticks/
  ├── TSLA_trades_20250303.parquet
  ├── AAPL_trades_20250304.parquet
  └── ...

reports/
  ├── backtest_TSLA_20250303.html
  ├── TSLA_20250303_equity_tick.png
  └── ...
```

## Next Steps

1. **Find volatile dates** using the scanner
2. **Run backtests** on those specific dates
3. **Review HTML reports** to understand trade logic
4. **Adjust parameters** in `config/settings.yaml`:
   - `vwap_extension_threshold` (default: 1.15)
   - `volume_exhaustion_factor` (default: 0.6)
   - `max_portfolio_risk_percent` (default: 1.0)

5. **Validate strategy** across multiple symbols/dates
6. **Deploy to paper trading** once satisfied

## Key Files

| File | Purpose |
|------|---------|
| `src/backtest/historical_tick_fetcher.py` | Fetches tick data from Alpaca |
| `src/backtest/tick_backtest_engine.py` | Tick-level backtesting logic |
| `src/backtest/backtest_engine.py` | Bar-based backtesting (fallback) |
| `src/backtest/visualizer.py` | Charts and HTML reports |
| `test_tick_backtest.py` | Test script |

## Performance Notes

- **Tick data is large**: A liquid stock can have 100k+ trades per day
- **First fetch is slow**: Alpaca API takes time for large requests
- **Caching is automatic**: Subsequent runs use local cache (instant)
- **Memory efficient**: Uses Polars streaming for large datasets

## Troubleshooting

**No trades executed?**
- Check that date had parabolic movement (>80% gain)
- Verify VWAP extension > 115% at entry time
- Check execution window (10:00-11:00 AM ET)

**Slow data fetch?**
- First time = downloading from Alpaca
- Use `use_cache=True` (default)
- Narrow time window if possible

**Connection errors?**
- Verify API credentials in `.env` file
- Check internet connection
- Alpaca free tier has rate limits

---

**Ready to backtest!** Run `python test_tick_backtest.py` to start.
