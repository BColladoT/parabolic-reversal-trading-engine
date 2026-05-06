# Why Only 2 Setups Were Scanned

## The Problem

**26 parabolic setups found** → **2 passed filtering** = **92% filtered out!**

## Why So Many Filtered?

### 1. Days Up Filter (Biggest Issue)

| Symbol | Days Up | Required | Status |
|--------|---------|----------|--------|
| GME 2021-01-13 | 1 | 2-5 | ❌ Filtered |
| KOSS 2021-01-27 | 1 | 2-5 | ❌ Filtered |
| AMC 2021-06-02 | 2 | 2-5 | ✅ PASSED |
| OCGN 2021-02-08 | 2 | 2-5 | ✅ PASSED |

**Most parabolic moves happen on Day 1**, not after 2+ days!

### 2. Prior 5-Day Gain Filter

| Symbol | Prior Gain | Required | Status |
|--------|------------|----------|--------|
| SOFI 2021-01-07 | -2.5% | 30%+ | ❌ Filtered |
| GME 2021-01-13 | 10.4% | 30%+ | ❌ Filtered |
| OCGN 2021-02-08 | 414% | 30%+ | ✅ PASSED |

**Many parabolic moves are sudden news events** without prior buildup!

### 3. Price Filter

| Symbol | Price | Required | Status |
|--------|-------|----------|--------|
| OCGN 2020-12-22 | $0.80 | $1-50 | ❌ Filtered |
| KOSS 2021-01-27 | $57.41 | $1-50 | ❌ Filtered |
| GME 2021-01-22 | $64.86 | $1-50 | ❌ Filtered |

**Price limits exclude sub-$1 and over-$50 stocks**

## The 2 Setups That Passed

1. **OCGN 2021-02-08**: +70.7%, 2 days up, prior +414%, $15.84
2. **DIDI 2022-03-18**: +58.9%, 4 days up, prior +36%, $2.55

## Solutions

### Option 1: Test ALL 26 Setups (Recommended)

```bash
python run_all_setups.py
```

This bypasses the strict filtering and tests every setup.

### Option 2: Relax the Filtering Criteria

Edit `config/settings.yaml`:

```yaml
# Relaxed criteria for more setups
screening:
  consecutive_green_days: 1  # Allow day 1 moves
  min_prior_gain_percent: 0  # Allow sudden news pops
  min_price: 0.50            # Allow sub-$1 stocks
  max_price: 100.00          # Allow higher-priced moves
```

### Option 3: Test Individual Setups

```bash
# Test KOSS +232% day (was filtered out)
python run_historical_backtest.py --symbol KOSS --date 2021-01-27

# Test GME +55% day
python run_historical_backtest.py --symbol GME --date 2021-01-13

# Test AMC +71% day
python run_historical_backtest.py --symbol AMC --date 2021-06-02
```

## Notable Setups You Missed

| Symbol | Date | Gain | Why Filtered |
|--------|------|------|--------------|
| KOSS | 2021-01-27 | **+232%** | Day 1 only, price $57 |
| GME | 2021-01-13 | **+55%** | Day 1 only, prior gain 10% |
| GME | 2021-01-22 | **+52%** | Price $65, prior gain 7% |
| AMC | 2021-06-02 | **+71%** | ✅ PASSED |
| EXPR | 2021-01-22 | **+54%** | Day 1, prior gain -8% |

## Recommendation

**Run `python run_all_setups.py`** to test all 26 setups without strict filtering.

The strategy might work better on **Day 1 parabolic moves** than "First Red Day" multi-day setups!
