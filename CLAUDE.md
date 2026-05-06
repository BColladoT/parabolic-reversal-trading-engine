# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Parabolic Reversal Trading Engine: a quantitative system for fading parabolic price reversals (short-selling) in micro-cap equities. Two main subsystems:

1. **Live Trading Engine** (`src/main_engine.py` + `run.py`) - Event-driven intraday trading via Alpaca Markets API
2. **RL Training Pipeline** (`src/rl/` + `src/scripts/`) - Reinforcement learning (SAC) with Walk-Forward Optimization to learn optimal trade sizing

The strategy targets stocks with 60-500% intraday gains showing volume exhaustion, shorting when price extends >120% above VWAP and covering at mean reversion.

## Commands

### Environment Setup
```bash
# Windows venv (for live trading + backtesting)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# WSL venv (for RL training - requires Ray/RLlib + PyTorch)
python -m venv venv_wsl
source venv_wsl/bin/activate
pip install -r wfo_context/requirements.txt
```

### Running
```bash
python run.py                                          # Live paper trading
python run_historical_backtest.py --quick-test          # Quick backtest (10 setups)
python run_historical_backtest.py --symbol AMC --date 2021-06-02  # Single setup
python scan_extended_universe.py                        # Scan 3,527 symbols for setups
```

### RL Training (WSL)
```bash
export PYTHONPATH=/mnt/c/quant_trading/src:$PYTHONPATH
python src/scripts/train_wfo_quick_test.py              # Quick test (~1-2 hours, 1 fold)
python src/scripts/train_wfo.py                         # Full WFO training (8+ hours, multiple folds)
python src/scripts/behavioral_cloning.py                # BC pretraining for SAC initialization
python src/scripts/compare_rl_vs_rule.py                # RL vs rule baseline comparison
```

### Testing
Tests are standalone scripts (no pytest framework):
```bash
python test_engine.py                # Core engine component validation
python test_connection.py            # Alpaca API connectivity
python test_quick_train.py           # RL training pipeline smoke test
python test_env_accounting.py        # RL environment accounting invariants
python test_wfo_data_leakage_complete.py  # WFO data leakage prevention
```

## Architecture

### Live Trading Data Flow
```
WebSocket ticks (AlpacaClient)
  -> Bar aggregation (PolarsSignalEngine, StreamingBuffer with deque)
  -> Numba JIT indicators (VWAP, ATR, volume profiles)
  -> Signal evaluation (ParabolicSignalEngine - needs >=2 exhaustion factors)
  -> Risk sizing (RiskManager - 1% max risk, 3-tier scale-in)
  -> Order execution (AlpacaClient REST)
```

Event-driven via callbacks: `AlpacaClient.tick_callback -> TradingEngine._on_tick()` and `SignalEngine.register_callback -> TradingEngine._on_signal()`.

### RL System Architecture
```
Parquet 1-min bars (HybridDataProvider: 70% CSV winners + 30% all high-vol days)
  -> Feature engineering (VWAP deviation, volume concentration)
  -> TCN-AE perception (60-bar OHLCV -> 64-dim latent, causal convolutions)
  -> 74-dim observation vector [latent(64) + explicit features(10)]
  -> SAC agent (continuous action in [-1, 1] = target short exposure)
  -> ParabolicReversalEnv (Gymnasium, Sortino reward + quadratic drawdown penalty)
```

**Walk-Forward Optimization**: Rolling 2-year train / 6-month test windows with 10-day purge. Two-phase per fold: Phase 1 freezes Actor (critic warm-up, 20K steps), Phase 2 unfreezes Actor (fine-tuning, 50K steps). BC weights initialize the Actor.

### Key Module Locations

| Concern | Location |
|---------|----------|
| Live orchestrator | `src/main_engine.py` (TradingEngine) |
| Broker API | `src/data/alpaca_client.py` (AlpacaClient) |
| Data processing | `src/data/polars_engine.py` (PolarsSignalEngine) |
| JIT indicators | `src/indicators/numba_kernels.py` (@njit VWAP/ATR/absorption) |
| Signal generation | `src/execution/signal_engine.py` (ParabolicSignalEngine) |
| Risk management | `src/risk/position_manager.py` (RiskManager) |
| Strategy registry | `src/strategies/strategy_registry.py` (V5 relaxed = recommended) |
| RL environment | `src/rl/env.py` (ParabolicReversalEnv) |
| RL data provider | `src/rl/data_provider_hybrid.py` (HybridDataProvider) |
| RL config | `src/rl/config.py` (RL_CONFIG circuit breakers) |
| WFO training | `src/scripts/train_wfo.py` |
| BC pretraining | `src/scripts/behavioral_cloning.py` |
| Rule baseline | `src/baselines/rule_baseline.py` |
| Shared metrics | `src/utils/metrics.py` (compute_fold_metrics - single source of truth) |
| Config loading | `src/utils/config.py` (YAML -> nested dataclasses -> global CONFIG) |

### wfo_context/ Directory
Self-contained copy of the RL training code for isolated WFO experiments. Mirrors `src/rl/`, `src/baselines/`, `src/scripts/`, and `src/utils/metrics.py` with its own `config/settings.yaml`. Used to run training without risk of modifying the main codebase.

## Code Conventions

- **Polars, not Pandas**: Use Polars DataFrames and LazyFrames exclusively. The system targets 30-50x speedup over Pandas.
- **Numba for numerics**: All indicator calculations use `@njit(cache=True, fastmath=True)`. Convert Polars/NumPy to NumPy arrays before Numba calls.
- **Structured logging**: Use `structlog` / `python-json-logger`. No print statements in production code.
- **Dataclass state**: All domain objects (`Position`, `TradeSignal`, `BarData`, `TickData`, etc.) are Python dataclasses.
- **Configuration**: Loaded from `config/settings.yaml` via nested dataclasses in `src/utils/config.py`. Global singleton `CONFIG`. Broker credentials come from `.env` via `os.getenv()`.

## Critical Constraints

- **No overnight positions**: All positions flatten by 15:25 ET. The system is strictly intraday.
- **WFO data leakage**: `HybridDataProvider` accepts `date_range=(start, end)` and has runtime assertions preventing cross-fold contamination. Each fold must create isolated provider instances.
- **Causal-only data access**: TCN-AE uses causal convolutions. `get_pre_decision_sequence()` excludes current bar. Observations only use bars at or before the decision point.
- **Circuit breakers**: Max single trade loss -$19,180, max drawdown -$19,180, max position value $30K, quarter-Kelly sizing. These are empirical limits from V5 Relaxed backtests.
- **VWAP anchored at 9:30 AM ET**: Session VWAP resets daily. All extensions calculated relative to this anchor.

## Data

- Historical data cached as Parquet files in `data/cache/` (1-min bars in `data/cache/1min_extended/`). First backtest run downloads data; subsequent runs use cache.
- CSV setup files in `reports/` contain proven backtest winners (symbol, date, PnL) used by the hybrid data provider.
- Models saved in `models/` (PyTorch `.pt` files for BC, RLlib checkpoints for WFO folds).

## Environment Variables

Required in `.env` (see `.env.template`):
```
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET=your_secret_here
```

For RL training in WSL, also set: `export PYTHONPATH=/mnt/c/quant_trading/src:$PYTHONPATH`
