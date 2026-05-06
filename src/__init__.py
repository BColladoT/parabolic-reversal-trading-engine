"""
Parabolic Reversal Trading Engine
High-performance quantitative trading system for fading blow-off tops.

Components:
- data: Alpaca WebSocket streaming + Polars data engine
- indicators: Numba-optimized VWAP, ATR calculations
- screening: Asset qualification and filtering
- risk: Position sizing and risk management
- execution: Signal generation and order routing
"""

__version__ = "1.0.0"
__author__ = "Quant Trading Team"
