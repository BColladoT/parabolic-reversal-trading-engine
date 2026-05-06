# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-06

### Added
- Initial private commit of the parabolic-reversal trading engine.
- Rules engine V5 Relaxed: multi-factor entry (VWAP extension > 120%, volume exhaustion < 60% of session peak, momentum/volume divergence, absorption detection), 3-tier scale-in (25 / 25 / 50%), layered exits (TP1/TP2/TP3 + trailing stop), 1% portfolio risk cap.
- RL position-sizing pipeline (research, in development): Soft Actor-Critic with causal TCN-autoencoder perception, behavioral-cloning warm-start, walk-forward optimization with 10-day purge gap, two-phase per fold.
- Live trading orchestrator (`src/main_engine.py`) with Alpaca SIP WebSocket ingestion, Polars bar aggregation, and Numba-JIT indicator kernels.
- Hybrid data provider (70% known winners + 30% all-volatility days) with runtime data-leakage assertions.
- Tick-level backtester modelling 2-tick slippage and $0.005/share commission.
- Sample dataset (5 symbols × 30-day windows, ~430 KB) and self-contained quickstart example backtest.
- Pytest test suite covering Numba indicator kernels and the quickstart smoke path.
- GitHub Actions CI workflow running pytest across Python 3.9 / 3.10 / 3.11.
- Walkthrough notebook (`notebooks/01_backtest_walkthrough.ipynb`) demonstrating the entry signal on KOSS 2021-01-28.
- Documentation: V5 Relaxed comprehensive technical report, equity curve and P&L distribution charts, Mermaid architecture diagrams in the README.
