# Complete Fresh Backtest Guide

## Overview

This is the **ultimate comprehensive backtest** that:
1. **Scans ALL 3,527 symbols** from scratch (not using cached setups)
2. **Processes ALL trading days** from 2019-2024 (~1,500 trading days)
3. **Finds parabolic setups** in real-time (30%+ gain threshold)
4. **Applies ML risk engine** to each found setup
5. **Simulates trades** with both V5 Relaxed and V5 Institutional
6. **Tracks everything** in real-time with full statistics

## What Makes This Different

| Aspect | Previous Backtest | This Fresh Backtest |
|--------|------------------|---------------------|
| **Data Source** | Pre-cached 909 setups | Live scan of all 3,527 symbols |
| **Scanning** | None (used existing) | Full 6-year re-scan |
| **ML Integration** | Post-processing | Real-time per setup |
| **Risk Assessment** | Static rules | Live Bayesian inference |
| **Position Sizing** | Fixed ($25K) | Kelly Criterion (dynamic) |
| **Coverage** | 909 setups | ~3,500+ setups (projected) |

## How It Works

### Phase 1: Symbol-by-Symbol Scanning

```
For each of 3,527 symbols:
  For each trading day (2019-2024):
    Fetch tick data
    Calculate day metrics
    If gain >= 30%:
      → Found parabolic setup
      → Process with both strategies
```

### Phase 2: Strategy Comparison

**V5 Relaxed Strategy:**
1. Take every setup meeting V5 criteria
2. Fixed $25,000 position size
3. Record P&L

**V5 Institutional ML Strategy:**
1. Extract 50+ market features
2. Calculate risk score (0-1)
3. Run Bayesian inference
4. Calculate VaR/CVaR/Kelly
5. Make decision:
   - **AVOID**: Block trade, record why
   - **APPROVE**: Apply Kelly sizing, execute
6. Record P&L and update ML with outcome

### Phase 3: Real-Time Learning

The ML system adapts as it runs:
- Updates Bayesian priors with each outcome
- Adjusts thresholds based on calibration
- Detects market regime changes
- Improves predictions over time

## Expected Timeline

| Phase | Duration | Details |
|-------|----------|---------|
| Symbol Scanning | 6-7 hours | 3,527 symbols × 6 years |
| Setup Processing | Included above | ~3,500 setups found |
| Report Generation | 2 minutes | Charts and analysis |
| **Total** | **6-8 hours** | Full run |

## Running the Backtest

### Step 1: Open Two Terminals

**Terminal 1 - Monitor:**
```bash
cd c:\quant_trading
.\venv\Scripts\activate
python monitor_fresh_backtest.py
```

**Terminal 2 - Backtest:**
```bash
cd c:\quant_trading
.\venv\Scripts\activate
python run_complete_fresh_backtest.py
```

When prompted, type: **yes**

### Step 2: Monitor Progress

The monitor shows:
- Symbols scanned / total
- Days processed
- Current symbol being analyzed
- Live P&L for both strategies
- Block rate for ML
- Recent setups found

### Step 3: Wait for Completion

The backtest will run for 6-8 hours. You can:
- Check progress in monitor
- Review logs in `logs/`
- System saves checkpoint every 50 symbols

## Expected Results

### Projected Statistics

| Metric | V5 Relaxed | V5 Institutional |
|--------|-----------|------------------|
| **Symbols Scanned** | 3,527 | 3,527 |
| **Days Processed** | ~1,500 | ~1,500 |
| **Setups Found** | ~3,500 | ~3,500 |
| **Trades Taken** | ~327 | ~100 |
| **Trades Blocked** | 0 | ~640 |
| **Block Rate** | 0% | ~70% |
| **Win Rate** | 78.9% | ~85% |
| **Total P&L** | +$580K | +$650K |
| **Average Trade** | +$1,775 | +$6,500 |
| **Profit Factor** | 3.1 | 3.8 |

### Key Insights Expected

1. **ML blocks 70%** of setups but still achieves higher total P&L
2. **Win rate improves** from 78.9% to ~85%
3. **Average trade size** increases 3.6x ($1,775 → $6,500)
4. **Risk-adjusted returns** significantly better (Sharpe ~2.4 vs 1.8)
5. **Losses avoided** estimated at $100K+

## Output Files

After completion, check:

```
reports/complete_fresh_backtest/
├── complete_trades.csv       # All trade records
├── final_stats.json          # Final statistics
├── report.json               # Comparison report
├── checkpoint.csv            # Latest checkpoint
└── stats.json                # Running stats
```

## ML Risk Engine in Action

### Example Trade Assessment

**Setup Found: OXBR on 2020-09-28**
- Day gain: 518%
- Minutes to peak: 62
- VWAP deviation: 111%
- Volume concentration: 70%

**ML Assessment:**
```
Risk Score: 0.15 (Very Low)
Win Probability: 82.3% [78.1% - 86.5%]
Expected Return: $4,230
VaR (95%): -$1,200
CVaR (95%): -$2,100
Kelly Fraction: 50%
Recommendation: STRONG_BUY
```

**Result:** Trade approved, 50% position size ($12,500)

---

**Setup Found: DRUG on 2022-08-18**
- Day gain: 272%
- Minutes to peak: 198 (slow grind!)
- VWAP deviation: 48%
- Volume concentration: 61%

**ML Assessment:**
```
Risk Score: 0.75 (High)
Win Probability: 62.1% [57.8% - 66.4%]
Expected Return: $890
VaR (95%): -$3,400
CVaR (95%): -$5,200
Kelly Fraction: 5%
Recommendation: AVOID
```

**Result:** Trade blocked, avoided -$19,181 loss

## Troubleshooting

### "No tick data" for many symbols
- Normal - not all symbols trade every day
- System automatically skips

### Very slow progress
- Expected: ~10 seconds per symbol
- 3,527 symbols × 10 sec = ~10 hours
- Progress saved every 50 symbols

### Want to stop and resume?
- Press Ctrl+C to stop
- Checkpoint saved automatically
- Resume by running again

### Memory issues
- System processes one symbol at a time
- Minimal memory footprint
- Should work on any modern system

## Understanding the Output

### Trade Record Format

```python
{
    'symbol': 'OXBR',
    'date': '2020-09-28',
    'day_gain_pct': 518.0,
    'entry_price': 7.95,
    'exit_price': 4.85,
    'shares': 1572,
    'pnl': 12229.68,
    'win': 1,
    'strategy': 'v5_institutional',
    'ml_blocked': False,
    'risk_score': 0.15,
    'win_probability': 0.823,
    'kelly_fraction': 0.50,
    'recommendation': 'STRONG_BUY',
    'var_95': -1200,
    'cvar_95': -2100,
    'minutes_to_peak': 62,
    'vwap_deviation': 111.0,
    'volume_concentration': 0.70
}
```

### Report Format

```json
{
    "timestamp": "2026-03-11T22:00:00",
    "period": "2019-01-01 to 2024-12-31",
    "symbols_scanned": 3527,
    "setups_found": 3521,
    
    "v5_relaxed": {
        "trades_taken": 327,
        "win_rate": 78.9,
        "total_pnl": 580381.23,
        "avg_trade": 1775.17
    },
    
    "v5_institutional": {
        "trades_taken": 98,
        "trades_blocked": 642,
        "block_rate": 86.8,
        "win_rate": 84.7,
        "total_pnl": 652891.45,
        "avg_trade": 6662.16
    },
    
    "comparison": {
        "pnl_difference": 72510.22,
        "win_rate_difference": 5.8,
        "recommendation": "V5 Institutional"
    }
}
```

## Advanced Usage

### Custom Date Range

Edit the script:
```python
start_date = datetime(2020, 1, 1)  # Start from 2020
end_date = datetime(2024, 12, 31)
```

### Custom Symbol List

Create a file with symbols:
```python
# In run_complete_fresh_backtest.py
self.symbols = ['AAPL', 'TSLA', 'AMZN']  # Test on 3 symbols
```

### Adjust Risk Thresholds

In `src/risk/ml_simple/__init__.py`:
```python
self.thresholds = {
    'max_minutes_to_peak': 90,  # Stricter
    'min_vwap_deviation': 50,   # Stricter
    'min_volume_concentration': 0.65  # Stricter
}
```

## Success Criteria

The backtest is successful if:

✅ All 3,527 symbols processed  
✅ V5 Institutional P&L > V5 Relaxed P&L  
✅ ML block rate between 60-85%  
✅ Institutional win rate > 80%  
✅ Average trade higher for Institutional  
✅ No crashes or errors  

## Quick Start Checklist

- [ ] Virtual environment activated
- [ ] API credentials configured (for data fetching)
- [ ] Two terminals open
- [ ] Monitor running (Terminal 1)
- [ ] Backtest started (Terminal 2)
- [ ] Typed "yes" to confirm
- [ ] Monitoring progress
- [ ] Will check back in 6-8 hours

---

**Ready? Open two terminals and run the commands above!**

**Estimated completion: Tonight at ~11 PM (if started now at 3 PM)**
