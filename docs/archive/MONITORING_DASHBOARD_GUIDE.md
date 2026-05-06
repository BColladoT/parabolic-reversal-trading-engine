# Backtest Monitoring Dashboard Guide

## Overview

The monitoring dashboard provides real-time visibility into the backtest progress, P&L, and strategy performance.

## Launching the Monitor

### Option 1: Simple Text Monitor (Recommended)
```bash
python monitor_backtest.py
```

### Option 2: Curses-Based Monitor (Unix/Linux/Mac)
```bash
python monitor_backtest.py
# Automatically detects and uses curses if available
```

## Dashboard Layout

```
================================================================================
BACKTEST MONITOR - 14:32:15
================================================================================

Processed: 450 setups

--- V5 RELAXED ---
  Trades: 89
  Win Rate: 78.7%
  Total P&L: $145,230
  Avg Trade: $1,632

--- V5 INSTITUTIONAL ---
  Trades: 34 | Blocked: 116 (77.3%)
  Win Rate: 82.4%
  Total P&L: $152,891
  Avg Trade: $4,497

--- COMPARISON ---
  P&L Difference: +$7,661
  Win Rate Difference: +3.7%

--- RECENT TRADES ---
  OXBR     V5   WIN    $12,230
  DRUG     ML   SKIP       $0
  XTKG     V5   LOSS  -$15,886
  XTKG     ML   SKIP       $0
  KXIN     V5   WIN   $10,703
  KXIN     ML   WIN   $10,703

================================================================================
Refreshing in 5 seconds... (Ctrl+C to exit)
```

## Metrics Explained

### V5 Relaxed Scanner
| Metric | Description |
|--------|-------------|
| Trades | Number of trades executed |
| Win Rate | Percentage of winning trades |
| Total P&L | Cumulative profit/loss |
| Avg Trade | Average P&L per trade |

### V5 Institutional ML
| Metric | Description |
|--------|-------------|
| Trades | Number of trades taken after ML filtering |
| Blocked | Number of trades rejected by ML risk engine |
| Block Rate | Percentage of setups filtered out |
| Win Rate | Win rate on approved trades only |

### Comparison
| Metric | Description |
|--------|-------------|
| P&L Difference | Institutional minus V5 (positive = ML wins) |
| Win Rate Difference | Difference in win rates |

### Recent Trades
Shows last 6 trades with:
- **Symbol**: Stock ticker
- **V5/ML**: Which strategy
- **WIN/LOSS/SKIP**: Outcome
- **$Amount**: P&L (0 for blocked trades)

## Key Indicators to Watch

### 1. Block Rate
- **Normal**: 60-80%
- **High**: >80% (very selective)
- **Low**: <50% (may be taking too much risk)

### 2. Win Rate Difference
- **Target**: +5% or more for ML
- **Good**: +3-5%
- **Concerning**: <0% (ML underperforming)

### 3. P&L Difference
- Should be **positive** for ML strategy
- If negative, ML may be filtering too aggressively

### 4. Average Trade Size
- ML should have **higher** average trade
- Shows better risk-adjusted selection

## Color Coding (Curses Mode)

```
Green  = Positive P&L / Win
Red    = Negative P&L / Loss  
White  = Blocked/Neutral
Yellow = Headers/Warnings
Blue   = ML Strategy stats
```

## Real-Time Alerts

The monitor will highlight:

### Trade Blocked
```
[2024-03-11 14:30:15] DRUG   | Risk: 0.85 | Decision: AVOID
```
- High risk score (>0.7)
- Low win probability (<60%)
- Slow grind detected

### Trade Approved
```
[2024-03-11 14:32:01] OXBR   | Risk: 0.25 | Win%: 82% | Decision: STRONG_BUY
```
- Low risk score (<0.3)
- High win probability (>75%)
- Explosive parabolic pattern

## Keyboard Controls

| Key | Action |
|-----|--------|
| `Ctrl+C` | Exit monitor |
| `q` | Quit (curses mode) |
| `r` | Force refresh (curses mode) |
| `p` | Toggle pause (curses mode) |

## Log Files

Monitor also writes to:
```
logs/backtest_comparison.log
```

View with:
```bash
# Real-time log
tail -f logs/backtest_comparison.log

# Search for specific symbol
grep "OXBR" logs/backtest_comparison.log

# Find all blocked trades
grep "BLOCKED" logs/backtest_comparison.log
```

## Troubleshooting

### Monitor Not Updating
1. Check if backtest is still running
2. Verify `reports/comparison_checkpoint.csv` exists
3. Restart monitor

### No Data Displayed
1. Ensure backtest has started
2. Wait for first checkpoint (every 50 setups)
3. Check file permissions

### Slow Refresh
- Normal during heavy I/O
- Monitor reads from disk every 5 seconds
- Consider reducing refresh interval

## Advanced Usage

### Custom Refresh Rate
```python
# In monitor_backtest.py, change:
time.sleep(5)  # Change to 1, 2, 10, etc.
```

### Filter Specific Symbols
```bash
# Monitor only specific symbols
python monitor_backtest.py --symbols "OXBR,KXIN,DRUG"
```

### Export to CSV
```python
# Save monitor output
python monitor_backtest.py --export monitor_log.csv
```

## Interpretation Guide

### Scenario 1: High Block Rate, Better P&L ✅
```
Block Rate: 75%
P&L Diff: +$50,000
```
**Interpretation**: ML is working correctly - filtering losers, keeping winners

### Scenario 2: Low Block Rate, Similar P&L ⚠️
```
Block Rate: 30%
P&L Diff: +$5,000
```
**Interpretation**: ML not selective enough - adjust thresholds

### Scenario 3: High Block Rate, Lower P&L ❌
```
Block Rate: 85%
P&L Diff: -$20,000
```
**Interpretation**: ML too aggressive - filtering good trades

## Best Practices

1. **Start Monitor First**
   ```bash
   # Terminal 1
   python monitor_backtest.py
   
   # Terminal 2
   python run_comprehensive_backtest.py
   ```

2. **Check Every 30 Minutes**
   - Verify block rate is reasonable (60-80%)
   - Ensure P&L difference is positive
   - Watch for any errors

3. **Save Screenshots**
   - Capture key milestones (25%, 50%, 75%, 100%)
   - Document any anomalies

4. **Review Blocked Trades**
   ```bash
   grep "AVOID" logs/backtest_comparison.log | head -20
   ```

## Example Session

```bash
# Terminal 1 - Start monitor
$ python monitor_backtest.py
BACKTEST MONITOR (Text Mode)
Monitoring... Press Ctrl+C to exit
Waiting for backtest to start...

# Terminal 2 - Start backtest
$ python run_comprehensive_backtest.py
# Enter "yes" when prompted

# Terminal 1 - Monitor updates
================================================================================
BACKTEST MONITOR - 14:32:15
================================================================================
Processed: 50 setups

--- V5 RELAXED ---
  Trades: 12
  Win Rate: 75.0%
  Total P&L: $18,450

--- V5 INSTITUTIONAL ---
  Trades: 4 | Blocked: 46 (92.0%)
  Win Rate: 100.0%
  Total P&L: $19,230

... (continues updating every 5 seconds)

# When backtest completes, generate report
$ python generate_comparison_report.py
$ start reports/comparison_report.html
```

---

**Quick Start**:
```bash
# Terminal 1
python monitor_backtest.py

# Terminal 2  
python run_comprehensive_backtest.py
# Type: yes
```
