# Strategy V2: Progressive Exhaustion Scale-In

## Overview

The strategy has been completely reimplemented based on your specifications:
- **No more "First Red Day"** — pure intraday parabolic fading
- **Volume Exhaustion** — enter when volume drops 40%+ from peak
- **Progressive Scale-In** — build position as exhaustion confirms
- **Layered Profit Taking** — 35% at VWAP, 35% at -8%, 30% at -15%
- **No Leverage (for now)** — parameter ready for future activation
- **Tight Stops** — 4% on initial, 3.5% on full position

---

## Key Changes from V1

| Aspect | Old Strategy | New Strategy |
|--------|--------------|--------------|
| **Setup Type** | First Red Day (multi-day) | Intraday exhaustion only |
| **Min Gain** | 80% | 60% |
| **Entry Trigger** | VWAP > 115% + volume < 60% | Volume < 60% of peak + near HOD |
| **Position Building** | Single entry | Scale-in (25% → 25% → 50%) |
| **Stop Loss** | ATR-based or apex | 4% initial, 3.5% averaged |
| **Profit Targets** | VWAP only | VWAP (35%), -8% (35%), -15% (30%) |
| **Leverage** | None | Disabled (parameter ready) |
| **Daily Loss Limit** | None | -2% of account |
| **Time Window** | 10:00-11:00 AM | 09:45 AM - 02:30 PM |
| **Flatten Time** | 15:45 | 15:25 |

---

## Strategy Logic

### 1. Scanner (Who to Watch)

```
QUALIFICATION CRITERIA:
├── Intraday gain > 60% from open
├── Price > $2.00
├── Volume > 500,000 shares
├── Shortable (ETB)
└── Time > 09:45 AM
```

**No multi-day requirement** — we trade stocks that went parabolic TODAY.

### 2. Entry Signal (Volume Exhaustion)

```
ENTRY CONDITIONS (ALL required):
├── Volume has dropped to < 60% of day's peak
├── Price within 5% of day's high
├── VWAP extension > 20% (price > 120% of VWAP)
└── 9:45 AM - 2:30 PM ET only
```

**The Logic**: When volume drops but price stays elevated, buying pressure is exhausted. We fade that exhaustion.

### 3. Progressive Scale-In (Building the Position)

```
POSITION BUILDING:

Add #1 (Initial Entry)
├── Trigger: Volume < 60% of peak
├── Size: 25% of max position ($7,500 of $30K max)
├── Stop: Entry + 4%
└── Action: ENTER

Add #2 (Confirmation)
├── Trigger: NEW HIGH on EVEN LOWER volume (< 50% of peak)
├── Size: 25% additional ($7,500)
├── Stop: Updated to average + 3.5%
└── Action: ADD

Add #3 (Full Conviction)
├── Trigger: Another new high on < 40% volume OR clear rejection
├── Size: 50% additional ($15,000)
├── Stop: Average + 3.5%
└── Action: ADD (Full Position)
```

**Why Scale-In?**
- Test the waters with small size first
- Add only when thesis confirms (new high on lower volume)
- Full size only on strong confirmation
- Reduces risk of entering too early

### 4. Exit Strategy (Layered Profit Taking)

```
EXIT LEVELS:

TP1: VWAP Mean Reversion (35% of position)
├── Trigger: Price hits VWAP
├── Shares: 35% of total
├── Action: Close 35%
└── Move stop to breakeven

TP2: Momentum Continuation (35% of position)
├── Trigger: Price down 8% from entry
├── Shares: 35% of total
└── Action: Close 35%

TP3: Final Target (30% of position)
├── Trigger: Price down 15% from entry OR 3:25 PM
├── Shares: Remaining 30%
└── Action: Close all

STOP LOSS (Emergency)
├── Trigger: Price hits stop (4% initial, 3.5% avg)
├── Action: Close 100% immediately
└── Reason: Thesis invalidated
```

### 5. Risk Management

| Rule | Value | Purpose |
|------|-------|---------|
| **Per-Trade Risk** | 1% of account at full size | Consistent risk per trade |
| **Initial Stop** | 4% from entry | Tight stop on test position |
| **Average Stop** | 3.5% from average entry | Tighter as we add |
| **Daily Loss Limit** | -2% of account | Stop trading if having bad day |
| **Max Positions** | 3 concurrent | Prevent over-concentration |
| **Max Adds** | 3 per position | Limit scaling |
| **Flatten Time** | 3:25 PM ET | No overnight risk |

---

## Configuration Parameters

All settings are in `config/settings.yaml`:

### Volume Exhaustion
```yaml
volume_exhaustion:
  entry_threshold: 0.60      # Volume < 60% of peak = entry
  add2_threshold: 0.50       # Volume < 50% = add #2
  add3_threshold: 0.40       # Volume < 40% = add #3
  price_proximity_to_high: 0.95  # Must be within 5% of HOD
  new_high_required_for_add: true
```

### Position Scaling
```yaml
scaling:
  initial_size_percent: 25   # First entry: 25%
  add2_size_percent: 25      # Second: 25%
  add3_size_percent: 50      # Third: 50%
  max_position_value: 30000  # Max $30K per position (no leverage)
```

### Exit Targets
```yaml
exits:
  tp1_percent: 35            # Close 35% at VWAP
  tp2_percent_drop: 8.0      # Close 35% at -8%
  tp3_percent_drop: 15.0     # Close 30% at -15%
```

### Risk Settings
```yaml
risk:
  initial_stop_percent: 4.0   # 4% stop on first entry
  average_stop_percent: 3.5   # 3.5% stop on full position
  daily_loss_limit_percent: 2.0  # Stop after -2% day
```

---

## Example Trade Walkthrough

**Stock $XYZ: Opens $5.00, runs to $10.00 (100% gain)**

```
09:45 AM - Scanner detects $XYZ (up 100%, qualifies for monitoring)
         - Peak volume: 2M shares in first 15 min
         - Tracking begins

10:15 AM - Volume drops to 1M (50% of peak)
         - Price: $10.20 (98% of HOD $10.40)
         - VWAP: $8.00 (27% extension)
         - ✅ ADD #1: Short 750 shares @ $10.20 = $7,650
         - Stop: $10.61 (+4%)

10:45 AM - Price makes NEW HIGH $10.80
         - Volume: 800K (40% of peak - even lower!)
         - ✅ ADD #2: Short 750 shares @ $10.80 = $8,100
         - New average: $10.50, 1,500 shares
         - Stop: $10.87 (+3.5%)

11:30 AM - Price hits $11.00 (new high) on 600K volume
         - Clear rejection wick forms
         - ✅ ADD #3: Short 1,500 shares @ $10.90 = $16,350
         - Final position: 3,000 shares @ $10.60 avg = $31,800
         - Stop: $10.97 (+3.5%)

12:00 PM - Price hits VWAP $8.50
         - ✅ TP1: Close 1,050 shares (35%) @ $8.50
         - Profit: $2,205
         - Stop moved to breakeven ($10.60)

01:30 PM - Price at $9.60 (-9.4% from entry)
         - ✅ TP2: Close 1,050 shares (35%) @ $9.60
         - Profit: $1,050

02:45 PM - Price at $9.00 (-15.1% from entry)
         - ✅ TP3: Close 900 shares (30%) @ $9.00
         - Profit: $540

TRADE RESULT:
Total Profit: $3,795
Return: 11.9% on $31,800 position
Account Return: 7.6% ($3,795 / $50,000)
Win/Loss: Full Win (all targets hit)
```

---

## Files Modified

| File | Changes |
|------|---------|
| `config/settings.yaml` | New strategy parameters |
| `src/utils/config.py` | Added VolumeExhaustionConfig, ScalingConfig, ExitsConfig, LeverageConfig |
| `src/execution/signal_engine.py` | Complete rewrite for volume exhaustion + scale-in logic |
| `src/risk/position_manager.py` | Progressive position building + layered exits |
| `src/screening/screener.py` | Removed multi-day requirement, pure intraday |
| `src/backtest/backtest_engine.py` | New exit logic (TP1/TP2/TP3) + scale-in simulation |

---

## How to Test

### 1. Backtest Specific Setup
```bash
python run_historical_backtest.py --symbol AMC --date 2021-06-02
```

### 2. Quick Test (10 setups)
```bash
python run_historical_backtest.py --quick-test
```

### 3. Full Backtest
```bash
python run_historical_backtest.py --full
```

---

## Future Enhancements (Ready to Implement)

### 1. Leverage Activation
To enable leverage, change in `config/settings.yaml`:
```yaml
leverage:
  enabled: true
  max_leverage: 3.0  # or 2.0
```

### 2. Add-to-Losers Logic
Currently only adds on new highs. To add when price drops but volume drops more:
```yaml
volume_exhaustion:
  add_on_pullback: true  # Add even if price drops, if volume drops more
```

### 3. Trailing Stops
After TP1, automatically trail stop to lock profits.

---

## Key Advantages of V2

| Advantage | Why It Works |
|-----------|--------------|
| **Higher Win Rate** | Only entering on clear volume exhaustion |
| **Better Risk/Reward** | Tight 3.5% stops vs 10%+ profit targets |
| **No Overnight Risk** | Flat by 3:25 PM every day |
| **Flexible Sizing** | Scale in as conviction builds |
| **Captures Full Move** | Layered exits catch mean reversion and momentum |
| **Protected Downside** | -2% daily loss limit prevents blowups |

---

## Questions or Adjustments?

The strategy is fully implemented and ready to test. Common adjustments you might want:

1. **Tighter/Looser Stops** — Change `initial_stop_percent` and `average_stop_percent`
2. **Different Volume Thresholds** — Adjust `entry_threshold`, `add2_threshold`, `add3_threshold`
3. **Different Exit Levels** — Modify `tp2_percent_drop` and `tp3_percent_drop`
4. **Position Sizing** — Change `initial_size_percent`, `add2_size_percent`, `add3_size_percent`
5. **More/Less Adds** — Change `max_adds` (currently 3)

All parameters are in `config/settings.yaml`.

---

*Strategy Version: 2.0*  
*Implementation Date: 2026-03-10*  
*Status: Ready for Testing*
