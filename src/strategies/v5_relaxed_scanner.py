"""
V5 Relaxed Scanner Strategy

WINNING STRATEGY - Full 3,527 symbol backtest results:
- Setups Found: 909 (vs 242 original)
- Trades Taken: 327 (vs 40 original)
- Win Rate: 78.9% (maintained!)
- Total P&L: +$580,381 (11x improvement)
- Trades/Year: ~54 (vs 7 original)

This strategy uses:
- Relaxed DISCOVERY: 30% gain threshold (was 50%)
- Relaxed DISCOVERY: 2x volume (was 3x)
- Relaxed DISCOVERY: Allow single-day parabolics
- Strict ENTRY: V5 criteria (2-of-3: VWAP>15%, Vol<70%, Prox>93%)

The key insight: Relax DISCOVERY to find more setups, keep ENTRY strict
to maintain win rate. This captures 8x more trades with the same edge.

Usage with Scanner:
    from src.strategies.v5_relaxed_scanner import RelaxedScannerV5
    from src.backtest.historical_screener import HistoricalParabolicScreener
    
    # Scan with RELAXED criteria
    screener = HistoricalParabolicScreener()
    setups = screener.scan_for_parabolic_setups(
        symbols=symbols,
        start_date=start,
        end_date=end,
        min_gain_percent=30.0,      # Relaxed: 30% (was 50%)
        min_volume_multiplier=2.0,  # Relaxed: 2x (was 3x)
    )
    
    # Backtest with V5 strict entry
    engine = RelaxedScannerV5()
    for setup in setups:
        result = engine.run_tick_backtest(setup.symbol, setup.date)
"""

# This strategy uses the same V5 engine - the difference is in SCANNER parameters
# We import from v5_strict to make it clear they share the same entry logic
from .v5_strict import TickBacktestEngineV5, tick_backtest_engine_v5


class RelaxedScannerV5(TickBacktestEngineV5):
    """
    V5 with Relaxed Scanner Configuration.
    
    This is the WINNING strategy from full backtest.
    Uses V5 strict entry criteria but with relaxed scanner discovery.
    
    Entry Criteria (STRICT - same as V5):
        - 2-of-3 criteria met:
          1. VWAP extension > 15%
          2. Volume < 70% of peak
          3. Price within 7% of HOD
        - Stock up > 40% from open (was 50% in original)
        - Price > VWAP (momentum intact)
    
    Discovery Criteria (RELAXED):
        - Min gain: 30% (was 50%)
        - Min volume: 2x average (was 3x)
        - Allow single-day parabolics (was 2+ days)
        - Price range: $0.20-$100
    
    Results (3,527 symbols, 2019-2024):
        - 909 setups found (vs 242 original)
        - 327 trades taken (vs 40 original)
        - 78.9% win rate (maintained)
        - +$580,381 total P&L (11x improvement)
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        super().__init__(initial_capital)
        # Note: The actual relaxed parameters are in the SCANNER, not here
        # This engine maintains STRICT entry criteria
        # Use HistoricalParabolicScreener with relaxed params for discovery
        
    def get_strategy_info(self):
        """Return strategy information."""
        return {
            'name': 'V5 Relaxed Scanner',
            'status': 'RECOMMENDED',
            'win_rate': '78.9%',
            'total_pnl': '+$580,381',
            'trades': 327,
            'discovery': {
                'min_gain': '30% (was 50%)',
                'min_volume': '2x avg (was 3x)',
                'min_days_up': '1 (was 2)'
            },
            'entry': {
                'criteria': '2-of-3',
                'vwap_extension': '> 15%',
                'volume': '< 70% of peak',
                'proximity': '> 93% of HOD'
            }
        }


# Singleton instance
relaxed_scanner_v5 = RelaxedScannerV5()
