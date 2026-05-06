---
name: Parabolic Reversal Execution Agent
description: |
  Activate this skill when tasked with evaluating micro-cap equities exhibiting parabolic price advances, 
  volume exhaustion, or VWAP divergence. This agent utilizes Alpaca Markets infrastructure, Polars dataframes, 
  and Numba JIT compilation to execute the First Red Day and Blow-Off Top fading strategies.
  
  Key capabilities:
  - Real-time streaming via Alpaca WebSocket (SIP feed)
  - Numba-optimized VWAP and ATR calculations
  - Polars-based zero-copy data processing
  - Volatility-based position sizing
  - Spain tax compliance (Homogeneous Loss Rule)
  - Self-healing error recovery
---

# Parabolic Reversal Trading Engine

## Overview

This skill enables autonomous execution of a quantitative short-selling strategy targeting parabolic price reversals in micro-cap equities. The system is optimized for sub-millisecond signal generation using Polars DataFrames and Numba JIT compilation.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Trading Engine                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Alpaca    │  │   Polars    │  │       Numba         │ │
│  │  WebSocket  │──│ Data Engine │──│    Kernels          │ │
│  │   Client    │  │  (Streaming)│  │ (VWAP/ATR)          │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│         │                  │                     │          │
│         ▼                  ▼                     ▼          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  Screener   │  │   Signal    │  │   Risk Manager      │ │
│  │  (Qualify)  │  │   Engine    │  │ (Position Sizing)   │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│                                            │                │
│                                            ▼                │
│                                   ┌─────────────────────┐   │
│                                   │  Order Execution    │   │
│                                   └─────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Strategy Logic

### Entry Criteria
1. **VWAP Extension**: Price > 115% of VWAP (indicates overextension)
2. **Volume Exhaustion**: Current volume < 60% of peak volume
3. **Momentum Divergence**: Price making new highs, volume decreasing
4. **Absorption Detection**: High volume at ask, price stalled (iceberg orders)

### Exit Criteria
1. **Profit Target**: Cover at VWAP (mean reversion)
2. **Stop Loss**: ATR-based stop or parabolic apex (whichever is tighter)
3. **Time-Based**: Flatten 15 minutes before market close (no overnight risk)
4. **Minimum Profit**: 10% depreciation from entry

### Position Sizing
```
Position Size = (Account Equity × Risk%) ÷ (Entry Price - Stop Loss)
```
- Max risk per trade: 1% of portfolio
- Max positions: 3 concurrent
- Max daily trades: 5

## Key Files

| File | Purpose |
|------|---------|
| `src/main_engine.py` | Main orchestrator with self-healing |
| `src/data/alpaca_client.py` | WebSocket + REST API client |
| `src/data/polars_engine.py` | High-performance data processing |
| `src/indicators/numba_kernels.py` | JIT-compiled VWAP/ATR |
| `src/screening/screener.py` | Asset qualification |
| `src/risk/position_manager.py` | Risk management |
| `src/execution/signal_engine.py` | Signal generation |

## Commands

### Start Trading
```bash
cd c:\quant_trading
venv\Scripts\python src\main_engine.py
```

### Run Screener Only
```python
from src.screening.screener import ParabolicScreener
from src.data.alpaca_client import AlpacaClient

client = AlpacaClient()
screener = ParabolicScreener(client)
# Screen symbols...
```

### Calculate Indicators
```python
from src.indicators.numba_kernels import calculate_vwap_numba
import numpy as np

vwap = calculate_vwap_numba(highs, lows, closes, volumes)
```

## Configuration

Edit `config/settings.yaml` to adjust:
- Risk parameters (position size, max trades)
- Signal thresholds (VWAP extension, volume exhaustion)
- Execution window (optimal: 10:00-11:00 AM ET)
- Compliance settings (Spain tax rules)

## Environment Variables

```bash
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
```

## Self-Healing Behavior

The engine will automatically:
1. Reconnect WebSocket on disconnection
2. Retry failed orders (max 3 attempts)
3. Flatten positions on critical errors
4. Blacklist assets after losses (Spain compliance)
5. Log all errors to `logs/trading_engine.log`

## Risk Limits

| Parameter | Value |
|-----------|-------|
| Max Risk/Trade | 1% |
| Max Positions | 3 |
| Max Daily Trades | 5 |
| Min Account Equity | $2,000 |
| Flatten Time | 15:45 ET |

## Timezone Handling

- Local: Europe/Madrid (CET)
- Market: America/New_York (ET)
- Execution Window: 10:00-11:00 AM ET (16:00-17:00 CET)
- DST transitions handled automatically

## Compliance

### Spain Homogeneous Loss Rule
- Assets are blacklisted for 365 days after a realized loss
- Prevents repurchasing same asset for tax loss deduction
- Automatic enforcement via `ComplianceConfig`

### No Overnight Positions
- All positions must be closed by 15:45 ET
- Emergency flatten if connection lost near close

## Performance Optimization

- **Polars**: Multi-threaded, vectorized data processing
- **Numba**: C-level execution for indicators
- **WebSocket**: Real-time tick data via SIP feed
- **Zero-copy**: PyArrow → Polars memory transfer
- **LazyFrame**: Deferred computation for queries

## Logging

Structured JSON logs written to `logs/trading_engine.log`:
- All trades (entry/exit)
- Signal generation
- Risk checks
- Errors and reconnections

## Troubleshooting

### WebSocket Disconnection
- Check API credentials
- Verify SIP subscription
- Review rate limits

### Order Rejection
- Check shortable status
- Verify buying power
- Review margin requirements

### High Latency
- Reduce subscribed symbols
- Increase bar aggregation interval
- Check network connection

## References

- Strategy Document: `AI Agent Trading Terminal Manual Construction.docx`
- Alpaca API: https://alpaca.markets/docs/
- Polars Docs: https://pola.rs/
- Numba Docs: https://numba.pydata.org/
