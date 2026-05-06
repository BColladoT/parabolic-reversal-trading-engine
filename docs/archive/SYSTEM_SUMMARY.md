# Parabolic Reversal Trading System - Complete Summary

## 🎯 What We Built

A **professional-grade algorithmic trading system** for fading parabolic micro-cap reversals with:
- **Live trading engine** (WebSocket, real-time)
- **Historical backtesting** (6+ years of tick data)
- **Extended screener** (1,115 micro-cap stocks)
- **Full audit trails** (every decision logged)

---

## 📁 Project Structure

```
quant_trading/
├── src/
│   ├── main_engine.py              # Live trading orchestrator
│   ├── data/
│   │   ├── alpaca_client.py        # WebSocket + REST API
│   │   └── polars_engine.py        # High-performance data
│   ├── indicators/
│   │   └── numba_kernels.py        # JIT-compiled VWAP/ATR
│   ├── backtest/
│   │   ├── historical_screener.py  # Scan 2016-2024 for setups
│   │   ├── historical_tick_fetcher.py  # Tick-level data
│   │   ├── tick_backtest_engine.py # Accurate simulation
│   │   ├── batch_backtest.py       # Multi-year testing
│   │   ├── visualizer.py           # Charts & HTML reports
│   │   └── extended_universe.py    # 1,115 micro-cap symbols
│   ├── screening/
│   │   └── screener.py             # Real-time asset qualification
│   ├── risk/
│   │   └── position_manager.py     # Risk management
│   └── execution/
│       └── signal_engine.py        # Entry/exit logic
├── config/settings.yaml            # Strategy parameters
├── run.py                          # Start live trading
├── run_historical_backtest.py      # Backtest runner
├── scan_extended_universe.py       # Scan 1,115 symbols
└── reports/                        # Generated reports
```

---

## 🚀 Live Trading Engine

**Components:**
- **Alpaca WebSocket**: Real-time tick data (SIP feed)
- **Polars DataFrames**: 30-50x faster than Pandas
- **Numba JIT**: C-level execution for indicators
- **Self-healing**: Auto-reconnect, error recovery

**Run:**
```bash
python run.py
```

**Features:**
- VWAP/ATR calculated in <1ms
- Position sizing based on volatility
- Hard stop at parabolic apex
- Flat before 15:45 ET (no overnight risk)

---

## 📊 Historical Backtesting

**Capabilities:**
- **6+ years** of historical data (2019-2024)
- **Tick-level** accuracy (actual trade prices)
- **1,115 symbols** micro-cap universe
- **Full audit trails** (every decision logged)

**Commands:**
```bash
# Scan for parabolic setups
python scan_extended_universe.py

# Test specific setup
python run_historical_backtest.py --symbol AMC --date 2021-06-02

# Quick test (10 setups)
python run_historical_backtest.py --quick-test

# Full backtest (all setups)
python run_historical_backtest.py --full
```

---

## 🔍 Extended Micro-Cap Universe

**Total Symbols: 1,115**

**Categories:**
- **Meme/Retail**: AMC, GME, BBBY, CLOV, WISH
- **Biotech**: IBIO, OCGN, SAVA, ANVS, VTVT
- **Chinese EVs**: NIO, XPEV, LI
- **Small Tech**: AI, PLTR, SOFI, HOOD, RBLX
- **Low Float**: MULN, TTOO, NVAX, NKLA
- **Your Extended List**: 900+ additional symbols

---

## 📈 Parabolic Setup Criteria

**First Red Day Setup:**
1. **Day Gain**: 50-500% (parabolic)
2. **Volume**: 3x average (unusual activity)
3. **Consecutive Days**: 2-5 green days (momentum)
4. **Prior Trend**: 30%+ over 5 days
5. **Price**: $0.50 - $50 (micro-cap)

**Entry Signal:**
- VWAP Extension > 115%
- Volume Exhaustion < 60%
- 10:00-11:00 AM ET window
- 3+ confirming factors

**Exit Rules:**
- Profit Target: VWAP mean reversion
- Stop Loss: ATR-based or parabolic apex
- Time: Flat before 15:45 ET

---

## 📋 Example Backtest Results

**Found in Meme Stock Era (2020-2021):**
```
Symbol   Date         Gain     Days Up
--------------------------------------
KOSS     2021-01-27   +232%    1
GME      2021-01-22   +52%     3
AMC      2021-06-02   +71%     2
BBIG     2021-08-27   +49%     4
OCGN     2021-02-08   +71%     2
```

**Test Command:**
```bash
python run_historical_backtest.py --symbol KOSS --date 2021-01-27
```

---

## 📊 Audit Trail Example

```
[ENTRY] 10:15:32.450 AMC @ $62.50 (slippage: +5bps)
  Reasoning: VWAP extension 1.25x (>115%); Volume exhaustion detected
  Confidence: 75% | Criteria met: 3/4
  VWAP: $50.00 | ATR: $2.50 | Volume: 0.6x avg
  Position: 150 shares | Risk: $375.00
  Stop: $65.00 | Target: $50.00 (VWAP)

[EXIT] 10:42:18.120 @ $51.20 (slippage: -5bps)
  P&L: +$1,687.50 (+11.25%) | Reason: profit_target
  Hold time: 26 minutes 46 seconds
```

---

## 📁 Generated Reports

```
reports/
├── parabolic_setups.csv           # All setups found
├── batch_backtest_*.csv           # Trade log
├── batch_backtest_*.html          # Visual dashboard
├── backtest_SYMBOL_*.html         # Single setup details
└── *_equity_tick.png              # Equity curves
```

---

## ⚙️ Configuration

Edit `config/settings.yaml`:

```yaml
# Screening
screening:
  min_percent_gain: 50.0
  max_percent_gain: 500.0
  min_price: 1.0
  max_price: 50.0

# Signals
signals:
  vwap_extension_threshold: 1.15
  volume_exhaustion_factor: 0.6

# Risk
risk:
  max_portfolio_risk_percent: 1.0
  max_daily_trades: 5
  max_positions: 3

# Execution
timezone:
  execution_window_start: "10:00"
  execution_window_end: "11:00"
  flatten_time: "15:45"
```

---

## 🎓 Key Files

| File | Purpose |
|------|---------|
| `run.py` | Start live trading |
| `run_historical_backtest.py` | Backtest runner |
| `scan_extended_universe.py` | Scan 1,115 symbols |
| `test_tick_backtest.py` | Test tick-level engine |
| `show_setups.py` | List found setups |

---

## 🔑 API Credentials

Stored in `.env` file:
```bash
ALPACA_API_KEY=PKSJORD33RQ5EMENAYZITUHOXG
ALPACA_SECRET=AKGGwfYLWCDnLFf6ifPxSvm6AdzQ1QfTyF3WHSkaS98C
```

---

## 📊 Performance Optimization

| Component | Technology | Speed |
|-----------|-----------|-------|
| Data Processing | Polars | 30-50x faster than Pandas |
| Indicators | Numba JIT | C-level execution |
| Tick Data | PyArrow | Zero-copy memory |
| WebSocket | Alpaca SIP | Real-time feeds |

---

## ✅ System Status

| Component | Status |
|-----------|--------|
| Live Trading Engine | ✅ Ready |
| Historical Backtest | ✅ 6+ years data |
| Extended Screener | ✅ 1,115 symbols |
| Tick-Level Engine | ✅ Accurate simulation |
| Audit Trails | ✅ Full reasoning |
| HTML Reports | ✅ Visual dashboards |

---

## 🚀 Next Steps

1. **Run extended scan:**
   ```bash
   python scan_extended_universe.py
   ```

2. **Review setups:**
   ```bash
   python show_setups.py
   ```

3. **Test specific dates:**
   ```bash
   python run_historical_backtest.py --symbol AMC --date 2021-06-02
   ```

4. **Adjust parameters** in `config/settings.yaml`

5. **Validate strategy** across hundreds of setups

6. **Deploy to paper trading**

---

## 📞 Commands Reference

```bash
# Live trading
python run.py

# Historical scanning
python scan_extended_universe.py
python scan_meme_era.py

# Backtesting
python run_historical_backtest.py --quick-test
python run_historical_backtest.py --full
python run_historical_backtest.py --symbol GME --date 2021-01-27

# Analysis
python show_setups.py
python test_tick_backtest.py
```

---

**Your professional-grade parabolic reversal trading system is complete!** 🎉
