# Parabolic Reversal Trading Strategies

This folder contains all strategy implementations for the parabolic reversal trading system.

## Strategy Comparison

| Strategy | Entry Criteria | Win Rate | Total P&L | Status |
|----------|---------------|----------|-----------|--------|
| **V5 Strict** | 2-of-3 criteria, 50%+ gain | 80.0% | +$53,148 | Original |
| **V5 Relaxed Scanner** | 30% gain discovery, V5 entry | 78.9% | **+$580,381** | **RECOMMENDED** |
| **V6 Relaxed Entry** | 1-of-3 criteria | 13.3% | -$20,130 | Not Recommended |

## Recommended Strategy: V5 Relaxed Scanner

The winning combination is:
- **Discovery**: Relaxed scanner (30% gain, 2x volume, single-day allowed)
- **Entry**: V5 strict criteria (2-of-3: VWAP>15%, Vol<70%, Prox>93%)

This gives us **11x more profit** while maintaining the ~79% win rate.

## File Structure

```
src/strategies/
├── __init__.py              - Module exports
├── README.md                - This file
├── v5_strict.py             - Original V5 strict entry
├── v5_relaxed_scanner.py    - V5 with relaxed scanner (WINNING)
└── v6_relaxed_entry.py      - Relaxed entry (for reference)
```

## Usage

```python
# Import the winning strategy
from src.strategies.v5_strict import TickBacktestEngineV5

# Run backtest
engine = TickBacktestEngineV5()
result = engine.run_tick_backtest(symbol="AAPL", date=datetime(2024, 1, 15))

print(f"P&L: ${result.total_pnl:+.2f}")
print(f"Win Rate: {result.win_rate*100:.1f}%")
```

## Adding New Strategies

To add a new strategy:

1. Create a new file: `v7_your_strategy.py`
2. Inherit from base or copy an existing strategy
3. Modify the entry/exit criteria
4. Update `__init__.py` to export it
5. Document results here

## Strategy Development Guidelines

1. **Keep entry criteria strict** - This maintains win rate
2. **Relax discovery criteria** - This finds more opportunities  
3. **Test thoroughly** - Run on full 6-year dataset
4. **Document results** - Update this README with metrics

## Backtest Results (Full 3,527 Symbols)

### V5 Strict (Original)
- Setups: 242
- Trades: 40
- Win Rate: 80.0%
- Total P&L: +$53,148
- Trades/Year: ~7

### V5 Relaxed Scanner (RECOMMENDED)
- Setups: 909
- Trades: 327
- Win Rate: 78.9%
- Total P&L: +$580,381
- Trades/Year: ~54

### V6 Relaxed Entry (NOT RECOMMENDED)
- Setups: 909
- Trades: 45
- Win Rate: 13.3%
- Total P&L: -$20,130
- Result: Relaxing entry criteria kills profitability

## Key Insight

**Relax DISCOVERY, not ENTRY.**

The winning formula is:
- Find more parabolic setups (lower gain threshold)
- Apply strict entry criteria (maintains win rate)
- Capture 8x more trades with same edge
