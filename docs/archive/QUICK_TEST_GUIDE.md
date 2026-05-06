# Quick Test Guide (1-2 Hours)

## What This Test Does

This quick test runs a **shortened version** of the WFO training to verify everything works before committing to a full 6+ hour training run.

### Key Differences from Full Training

| Parameter | Quick Test | Full Training |
|-----------|-----------|---------------|
| Folds | 1 | 4 |
| Train Window | 6 months | 2 years |
| Test Window | 1 month | 6 months |
| Warmup Timesteps | 5,000 | 20,000 |
| Finetune Timesteps | 15,000 | 50,000 |
| Total Timesteps | 20,000 | 70,000 |
| Eval Episodes | 3 | 10 |
| Estimated Time | 1-2 hours | 6-8 hours |

## How to Run

### Option 1: Windows (Batch File)

Double-click or run in Command Prompt:
```cmd
run_quick_test.bat
```

### Option 2: WSL (Linux)

```bash
cd /mnt/c/quant_trading
chmod +x run_quick_test.sh
./run_quick_test.sh
```

### Option 3: Manual (More Control)

```bash
# In WSL
cd /mnt/c/quant_trading
source venv_wsl/bin/activate
cd src/scripts

# Run with defaults (1-2 hours)
python train_wfo_quick_test.py

# Or customize parameters
python train_wfo_quick_test.py \
    --warmup-steps 3000 \
    --finetune-steps 10000 \
    --train-months 3 \
    --test-months 1
```

## What to Look For

### ✅ Good Signs (Working Correctly)

```
INFO: HistoricalDataProvider: Scanned 3089 symbols in data/cache/1min_extended
INFO: HistoricalDataProvider: Index complete: 52347 trading days from 2984 symbols
...
INFO: Episode 1: AAPL 2023-05-15 (390 bars)
INFO: Episode 2: TSLA 2023-08-22 (405 bars)
...
==============================================================
TEST RESULTS FOR FOLD 1
==============================================================
Test Reward (PnL): $1,247.35     <-- NON-ZERO = GOOD!
Test Reward Max:   $3,891.22
Test Reward Min:   $-892.15
✅ Test reward is non-zero - data is loading correctly!
```

### ❌ Bad Signs (Not Working)

```
WARNING: Failed to load trading day, using default state
...
==============================================================
TEST RESULTS FOR FOLD 1
==============================================================
Test Reward (PnL): $0.00         <-- ZERO = BAD!
Test Reward Max:   $0.00
Test Reward Min:   $0.00
❌ CRITICAL: Test reward is ~$0.00 - data may not be loading!
```

## Troubleshooting

### If You See "$0.00 PnL"

1. **Check data provider first:**
   ```bash
   python test_data_provider.py
   ```

2. **Clear cache and retry:**
   ```bash
   python clean_data_cache.py
   python train_wfo_quick_test.py
   ```

3. **Verify Parquet files exist:**
   ```bash
   ls -la data/cache/1min_extended/ | head -20
   ```

4. **Check logs for errors:**
   ```bash
   cat models/wfo_test/quick_test_results.json
   ```

### If Test Passes (Non-Zero PnL)

You're ready for full training! Run:

```bash
cd src/scripts
python train_wfo.py
```

Or use the batch file:
```cmd
START_COMPLETE_BACKTEST.bat
```

## Expected Timeline

```
0:00 - Start
0:01 - Data provider scans 3089 files
0:03 - Index built: ~50,000 trading days found
0:05 - Training starts (Phase 1: Actor frozen)
0:25 - Phase 1 complete (5000 steps)
0:26 - Phase 2 starts (Actor unfrozen)
0:55 - Phase 2 complete (15000 steps)
1:00 - Evaluation starts (3 episodes)
1:10 - Evaluation complete
1:15 - Results saved
```

## Output Files

After the test completes, check these files:

1. **Results JSON:** `models/wfo_test/quick_test_results.json`
2. **Checkpoint:** `models/wfo_test/fold_1_checkpoint/`
3. **Logs:** Console output (no file log in quick test)

## Interpreting Results

### Test Reward (PnL) Meaning

| Value | Interpretation |
|-------|---------------|
| $0.00 | ❌ Data not loading - DO NOT run full training |
| <$100 | ⚠️ Very small PnL - may need more training steps |
| $100-$1000 | ✅ Reasonable for quick test - OK to proceed |
| >$1000 | ✅ Good performance - definitely ready |

**Note:** The exact PnL value doesn't matter much for the quick test. What matters is that it's **not $0.00**, which proves data is loading and the agent is learning.

## Next Steps

### If Test Passes

1. Run full training: `python train_wfo.py`
2. Or use: `START_COMPLETE_BACKTEST.bat`
3. Full training will take 6-8 hours

### If Test Fails

1. Check `test_data_provider.py` output
2. Verify Parquet files are in `data/cache/1min_extended/`
3. Check column names in Parquet files match expected format
4. Review error messages in console output
