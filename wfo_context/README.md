# WFO Training Context Package

This folder contains all files related to the Walk-Forward Optimization (WFO) RL training experiment.

## Folder Structure

```
wfo_context/
├── src/
│   ├── scripts/           # Training scripts
│   │   ├── train_wfo.py              # Main WFO training script (Ray RLlib SAC)
│   │   ├── train_wfo_quick_test.py   # Quick test version (1-2 hours)
│   │   └── compare_rl_vs_rule.py     # RL vs Rule Baseline comparison
│   ├── rl/                # RL environment and components
│   │   ├── env.py                    # ParabolicReversalEnv (Gymnasium env)
│   │   ├── data_provider_hybrid.py   # HybridDataProvider (CSV + Parquet)
│   │   ├── config.py                 # RL_CONFIG defaults
│   │   ├── agent.py                  # SAC agent components
│   │   └── perception.py             # TCN-AE encoder
│   ├── baselines/         # Benchmark baselines
│   │   ├── __init__.py
│   │   ├── rule_baseline.py          # RuleBasedAgent (V5 deterministic rules)
│   │   └── evaluate_baseline.py      # Evaluation harness
│   └── utils/             # Utilities
│       └── metrics.py                # compute_fold_metrics()
├── config/
│   └── settings.yaml      # Strategy parameters (VWAP thresholds, etc.)
├── reports/
│   └── relaxed_909_backtest.csv  # CSV setups (proven winning trades)
├── models/
│   └── wfo_first_real_benchmark/   # Training results (JSON outputs)
└── requirements.txt       # Python dependencies

## Data Files (Not Copied)

The following data files are used but NOT copied due to size:
- `data/cache/1min_extended/*.parquet` - 1-minute OHLCV bars for all symbols

To include these, manually copy from: `C:\quant_trading\data\cache\1min_extended\`

## Key Components

### 1. Training Pipeline
- `train_wfo.py`: Main training with 2-phase warmup (Actor frozen/unfrozen)
- Uses Ray RLlib SAC algorithm
- WFO splits: 1 year train / 3 months test / 10 days purge

### 2. Environment
- `env.py`: Custom Gymnasium environment
- State: 74-dim (64 TCN-AE latent + 10 explicit features)
- Action: Continuous [-1, 1] for position sizing
- Reward: True Sortino with drawdown penalty

### 3. Baselines
- Rule-based: V5 Relaxed deterministic strategy
- Comparison metrics: PnL, win rate, trades per episode
- Verdict: PASS/MARGINAL/FAIL based on 10% improvement threshold

### 4. Data Provider
- Hybrid: CSV setups (proven winners) + Parquet (all high-volatility days)
- Supports fixed_setup for deterministic evaluation
- Date range filtering for WFO (prevents data leakage)

## Running the Experiment

```bash
# 1. Training
cd /mnt/c/quant_trading
python3 -m src.scripts.train_wfo \
    --warmup-steps 5000 \
    --finetune-steps 10000 \
    --train-years 1 \
    --test-months 3 \
    --output-dir models/wfo_first_real_benchmark

# 2. Compare vs Rule Baseline
python3 -m src.scripts.compare_rl_vs_rule \
    --rl-results models/wfo_first_real_benchmark/wfo_results.json \
    --run-baseline \
    --output reports/first_real_comparison.json
```

## Dependencies

See `requirements.txt` for full list. Key packages:
- ray[rllib] 2.9.3
- torch
- numpy, polars, pyarrow
- gymnasium

## Results

Training outputs saved to:
- `models/wfo_first_real_benchmark/wfo_results.json`
- `models/wfo_first_real_benchmark/fold_1_checkpoint/`
- `reports/first_real_comparison.json`
