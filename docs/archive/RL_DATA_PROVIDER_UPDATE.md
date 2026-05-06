# RL Data Provider Update - All Parquet Files

## What Changed

The data provider (`src/rl/data_provider.py`) has been updated to use **ALL** available Parquet files instead of filtering for specific backtest setups.

### Before (Old Behavior)
- Loaded setups from `reports/relaxed_909_backtest.csv`
- Filtered for setups with `trades > 0` AND `gain_pct >= 60`
- Result: Only ~327 trading days available for training
- Problem: Model only trained on "successful" setups, no exposure to losing scenarios

### After (New Behavior)
- Scans all `*_1min_*.parquet` files in `data/cache/1min_extended/`
- Extracts **ALL** trading days from each file
- Result: ~50,000+ trading days available (estimated from 3,089 files)
- Benefit: Model trains on all market conditions - both winners and losers

## Key Features

1. **Automatic Index Caching**
   - First scan builds an index of all (symbol, date) pairs
   - Index cached to `data/cache/trading_days_index.pkl`
   - Subsequent runs load from cache (much faster)

2. **Parallel Scanning**
   - Uses ThreadPoolExecutor for faster initial scan
   - 8 workers process files concurrently

3. **Configurable Filtering**
   ```python
   date_range=("2020-01-01", "2024-12-31")  # Only dates in range
   min_bars_per_day=100                      # Skip incomplete days
   ```

4. **Comprehensive Logging**
   - Shows progress during scan
   - Reports total trading days found
   - Logs each episode load

## How to Test

Run the test script to verify everything works:

```bash
python test_data_provider.py
```

Expected output:
```
======================================================================
TESTING DATA PROVIDER - ALL PARQUET FILES
======================================================================

1. Initializing data provider...

2. Data Provider Statistics:
   - Files scanned: 3089
   - Symbols with data: ~3000+
   - Total trading days: ~50000+
   - Unique dates: ~1200+

3. Testing episode loading (loading 3 random trading days):
   Episode 1: AAPL on 2021-03-15 - 390 bars
   Episode 2: TSLA on 2020-07-22 - 405 bars
   Episode 3: GME on 2021-01-27 - 420 bars

✅ Data provider test complete!
   Total trading days available: ~50000
```

## How to Run Training

After confirming the test passes, run your WFO training:

```bash
cd src/scripts
python train_wfo.py
```

Or with custom parameters:
```bash
python train_wfo.py \
    --warmup-steps 20000 \
    --finetune-steps 50000 \
    --train-years 2 \
    --test-months 6
```

## Expected Improvements

With this change, you should see:

1. **More Diverse Training Data**
   - Model sees both winning and losing scenarios
   - Better generalization to unseen market conditions

2. **No More "Failed to load" Warnings**
   - All episodes will have real market data
   - No more $0.00 PnL from missing data

3. **Better Learning Signal**
   - Reward will reflect actual trading performance
   - Sortino ratio calculation will be meaningful

4. **Larger Training Dataset**
   - ~50,000 trading days vs ~327 before
   - 150x more training data

## Files Modified

- `src/rl/data_provider.py` - Complete rewrite to scan all Parquet files

## Files Added

- `test_data_provider.py` - Test script to verify data provider works
- `RL_DATA_PROVIDER_UPDATE.md` - This documentation

## Cache File

- `data/cache/trading_days_index.pkl` - Cached index of all (symbol, date) pairs
  - Generated automatically on first run
  - Delete this file to force re-scan if you add new Parquet files
