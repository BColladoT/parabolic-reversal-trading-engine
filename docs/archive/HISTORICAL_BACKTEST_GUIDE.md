# Historical Parabolic Reversal Backtest System

## Overview

We've built a **comprehensive historical backtesting system** that scans **6+ years of data (2019-2024)** to find micro-cap parabolic moves and tests the First Red Day strategy on each one.

## What It Does

### 1. Micro-Cap Universe

**Target Stocks:**
- Price: $0.50 - $50.00
- Market Cap: Under $500M
- Low float: < 100M shares preferred
- High volatility: Meme stocks, biotech, Chinese EVs

**Built-in Universe:**
```python
micro_cap_symbols = [
    # Low-float volatility
    'AMC', 'GME', 'BBBY', 'MULN', 'TTOO', 'NVAX', 'NKLA', 'PLUG',
    'BBIG', 'SPCE', 'TLRY', 'SOFI', 'HOOD', 'RBLX', 'PLTR',
    
    # Biotech (news-driven parabolic moves)
    'IBIO', 'VTVT', 'BIOC', 'OCGN', 'CVM', 'ATOS', 'SAVA', 'ANVS',
    
    # Chinese EVs
    'NIO', 'XPEV', 'LI',
    
    # Retail favorites
    'BB', 'NOK', 'KOSS', 'EXPR', 'CLOV', 'WKHS', 'WISH',
    
    # Small tech
    'AI', 'SOUN', 'IONQ', 'RGTI',
]
```

### 2. Parabolic Setup Detection

**Criteria for First Red Day Setup:**
1. **Single Day Gain**: 50-500% (parabolic but not insane)
2. **Volume**: 3x average daily volume
3. **Consecutive Days**: 2-5 green days (building momentum)
4. **Prior Trend**: 30%+ gain over previous 5 days
5. **Price**: $1-20 range (micro-cap sweet spot)

**Example Setup:**
```
Symbol: MULN
Date: 2023-03-15
Day Gain: +127%
Price: $0.52 → $1.18
Volume: 45M (5x average)
Days Up: 3 consecutive
Prior 5D Gain: +89%
→ QUALITY FIRST RED DAY SETUP
```

### 3. Tick-Level Backtesting

For each setup found, we:
1. **Fetch actual trade data** from Alpaca (every single trade)
2. **Calculate VWAP/ATR** from real ticks
3. **Simulate entry** when criteria met (10:00-11:00 AM window)
4. **Track position** tick-by-tick
5. **Execute exit** on stop loss, profit target, or time
6. **Record P&L** with realistic slippage (5bps)

## How to Use

### Quick Test (10 setups)
```bash
python run_historical_backtest.py --quick-test
```

### Full Historical Backtest (2019-2024)
```bash
python run_historical_backtest.py --full
```
**Note:** This will take 30+ minutes and test hundreds of setups.

### Scan and List All Setups
```bash
python run_historical_backtest.py --scan
```

Output:
```
TOTAL PARABOLIC SETUPS FOUND: 342

First 20 setups:
Date         Symbol   Gain     Price      Volume          Days Up
--------------------------------------------------------------------------------
2021-01-27   AMC      301.2%   $19.95     1,200,450,000   3
2021-01-28   GME      134.8%   $347.51    93,400,000      5
2023-06-15   MULN     127.4%   $1.18      45,200,000      3
2024-02-20   TTOO     89.5%    $3.45      28,100,000      4
...

SETUP ANALYSIS:
  Total Setups:           342
  Average Gain:           78.3%
  Median Gain:            65.4%
  Average Volume:         12,450,000
  Average Days Up:        3.2

Top 10 Most Frequent Symbols:
  MULN         23 setups
  AMC          18 setups
  TTOO         15 setups
  GME          12 setups
```

### Test Single Setup
```bash
python run_historical_backtest.py --symbol AMC --date 2021-01-27
```

## Backtest Results Include

### Performance Metrics
- **Win Rate**: % of trades profitable
- **Profit Factor**: Gross profit / gross loss
- **Average Trade**: Mean P&L per trade
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline

### Trade Analysis
- **Entry Reasoning**: Why each trade was taken
- **Confidence Score**: How many criteria met
- **Market Context**: VWAP, ATR, volume at entry
- **Hold Time**: Duration of each trade
- **Slippage**: Realistic execution costs

### Reports Generated
1. **CSV Trade Log**: Every trade with full details
2. **HTML Report**: Visual dashboard with charts
3. **Setup List**: All parabolic moves found
4. **Monthly Breakdown**: P&L by month/year

## Example Output

```
================================================================================
BATCH BACKTEST RESULTS (Multi-Year)
================================================================================

SETUP STATISTICS:
  Total Setups Scanned:     342
  Setups with Trades:       89
  Conversion Rate:          26.0%

TRADE STATISTICS:
  Total Trades:             156
  Winning Trades:           98
  Losing Trades:            58
  Win Rate:                 62.8%
  Profit Factor:            2.34

P&L STATISTICS:
  Total P&L:                $+47,250.00
  Avg P&L per Setup:        $+138.16
  Avg Return per Trade:     +1.24%
  Max Drawdown:             $-3,450.00
  Sharpe Ratio:             1.87

TOP PERFORMING SYMBOLS:
  AMC        $+12,450.00
  GME        $+8,320.00
  MULN       $+5,180.00
  TTOO       $+3,920.00
  NVAX       $+2,850.00
================================================================================
```

## Data Sources

### Alpaca Historical API
- **6+ years** of historical data
- **Tick-level** trades (actual execution prices)
- **Free tier**: IEX exchange data
- **Pro tier**: Full SIP (all exchanges)

### What's Available
- **Trades**: Every executed transaction
- **Quotes**: Bid/ask spreads
- **Bars**: 1Min/5Min/15Min/Daily
- **6+ years**: 2019-2024+

## Caching System

All data cached locally for speed:
```
data/cache/
├── ticks/
│   ├── AMC_trades_20210127.parquet
│   ├── GME_trades_20210128.parquet
│   └── ...
├── setups/
│   └── setups_20190101_20241231.pkl
└── bars/
    └── ...
```

## Configuration

Edit `config/settings.yaml`:

```yaml
# Signal thresholds
screening:
  min_percent_gain: 50.0      # Minimum parabolic gain
  max_percent_gain: 500.0     # Filter extreme outliers
  consecutive_green_days: 2   # First Red Day setup

# Risk management  
risk:
  max_portfolio_risk_percent: 1.0  # 1% risk per trade
  max_daily_trades: 5
  max_positions: 3

# Execution
signals:
  vwap_extension_threshold: 1.15   # Price > 115% of VWAP
  volume_exhaustion_factor: 0.6     # Volume < 60% of peak
```

## Key Insights to Look For

### Strategy Validation
1. **Win Rate > 50%**: Strategy edge confirmed
2. **Profit Factor > 1.5**: Profitable after costs
3. **Sharpe > 1.0**: Good risk-adjusted returns
4. **Max DD < 10%**: Manageable drawdown

### Pattern Analysis
1. **Which symbols work best?** AMC/GME style momentum
2. **Which gains work?** 50-150% better than 300%+
3. **Which days?** 2-3 days up better than 5+
4. **Which times?** 10:00-11:00 AM execution window

### Real-World Application
Once backtest validates the strategy:
1. Deploy to **paper trading**
2. Run **live screener** for real-time setups
3. Execute with **risk management**
4. Track **live performance** vs backtest

## Next Steps

1. **Run the scan** to see all historical setups:
   ```bash
   python run_historical_backtest.py --scan
   ```

2. **Run quick test** to validate:
   ```bash
   python run_historical_backtest.py --quick-test
   ```

3. **Full backtest** when ready:
   ```bash
   python run_historical_backtest.py --full
   ```

4. **Review reports** in `reports/` directory

5. **Adjust parameters** based on results

6. **Deploy to paper trading**

---

**Ready to validate your strategy across 6 years of micro-cap parabolic moves!** 🚀
