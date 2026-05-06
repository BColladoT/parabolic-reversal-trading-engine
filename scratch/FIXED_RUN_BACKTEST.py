#!/usr/bin/env python3
"""
FIXED Backtest Runner - Handles UTC timestamps correctly
"""
import sys
from datetime import datetime, time as dt_time
from pathlib import Path
import pytz

sys.path.insert(0, str(Path(__file__).parent))

from src.backtest.tick_backtest_engine import TickBacktestEngine
from src.indicators.numba_kernels import calculate_vwap_numba, calculate_atr_numba
from src.backtest.historical_tick_fetcher import tick_fetcher
from src.utils.config import CONFIG

import polars as pl


class FixedTickBacktestEngine(TickBacktestEngine):
    """Backtest engine with proper timezone handling."""
    
    def _in_execution_window(self, timestamp) -> bool:
        """Check if timestamp is in 10:00-11:00 AM ET execution window.
        
        Handles both UTC and ET timestamps.
        """
        # Convert to ET if needed
        if timestamp.tzinfo is not None:
            # Timestamp is timezone-aware (likely UTC)
            et_tz = pytz.timezone('America/New_York')
            timestamp = timestamp.astimezone(et_tz)
        
        t = timestamp.time() if isinstance(timestamp, datetime) else timestamp
        return dt_time(10, 0) <= t <= dt_time(11, 0)


# Use the fixed engine
fixed_engine = FixedTickBacktestEngine()


def run_fixed_backtest(symbol: str, date_str: str, verbose: bool = True):
    """Run backtest with timezone fix."""
    date = datetime.strptime(date_str, '%Y-%m-%d')
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"FIXED BACKTEST: {symbol} on {date_str}")
        print(f"Execution Window: 10:00-11:00 AM ET (timezone-aware)")
        print(f"{'='*80}\n")
    
    # Reset engine
    fixed_engine.reset()
    
    # Fetch tick data
    tick_df = tick_fetcher.fetch_combined_tick_data(symbol, date, use_quotes=False)
    
    if tick_df.is_empty():
        print(f"No tick data for {symbol} on {date_str}")
        return
    
    # Aggregate to bars
    bar_df = tick_fetcher.aggregate_trades_to_bars(tick_df, interval_seconds=60)
    
    # Calculate indicators
    highs = bar_df['high'].to_numpy()
    lows = bar_df['low'].to_numpy()
    closes = bar_df['close'].to_numpy()
    volumes = bar_df['volume'].to_numpy()
    
    vwap_values = calculate_vwap_numba(highs, lows, closes, volumes)
    atr_values = calculate_atr_numba(highs, lows, closes, period=14)
    
    bar_df = bar_df.with_columns([
        pl.Series('vwap', vwap_values),
        pl.Series('atr', atr_values),
        (pl.col('close') / pl.col('vwap')).alias('vwap_extension'),
    ])
    
    if verbose:
        print(f"Loaded {len(tick_df)} trades -> {len(bar_df)} bars")
    
    # Simulate tick-by-tick
    last_bar_time = None
    current_bar_data = None
    bars_in_window = 0
    
    for tick in tick_df.to_dicts():
        tick_time = tick['timestamp']
        
        # FIXED: Proper timezone handling
        if not fixed_engine._in_execution_window(tick_time):
            continue
        
        bars_in_window += 1
        
        # Update bar context
        tick_minute = tick_time.replace(second=0, microsecond=0)
        if tick_minute != last_bar_time:
            bar_match = bar_df.filter(pl.col('timestamp') == tick_minute)
            if len(bar_match) > 0:
                current_bar_data = bar_match.to_dicts()[0]
            last_bar_time = tick_minute
        
        if current_bar_data is None:
            continue
        
        # Get current price
        current_price = tick['trade_price']
        
        # Evaluate signals
        if fixed_engine.current_position is None:
            fixed_engine._evaluate_tick_entry(tick, current_bar_data, tick_df, verbose)
        else:
            fixed_engine._evaluate_tick_exit(tick, current_bar_data, verbose)
    
    # Force close at end
    if fixed_engine.current_position:
        last_tick = tick_df.to_dicts()[-1]
        fixed_engine._force_tick_exit(last_tick, "end_of_session", verbose)
    
    # Results
    result = fixed_engine._generate_result(symbol, date, date)
    
    if verbose:
        print(f"\n{'='*80}")
        print("RESULTS")
        print(f"{'='*80}")
        print(f"Bars in 10-11 AM window: {bars_in_window}")
        print(f"Total Trades: {result.total_trades}")
        print(f"Total P&L: ${result.total_pnl:+.2f}")
        
        if result.total_trades > 0:
            print(f"\nTRADE LOG:")
            for audit in result.audit_records:
                if audit.action.value == 'entry':
                    print(f"  ENTRY: {audit.timestamp.strftime('%H:%M:%S')} @ ${audit.price:.2f}")
                    print(f"    VWAP Ext: {audit.vwap_extension:.2f}x | Size: {audit.position_size}")
                elif audit.action.value == 'exit':
                    print(f"  EXIT:  {audit.timestamp.strftime('%H:%M:%S')} @ ${audit.exit_price:.2f}")
                    print(f"    P&L: ${audit.pnl:+.2f} | Reason: {audit.exit_reason}")
        else:
            print(f"\nNo trades executed.")
            print(f"Check if VWAP extension > {CONFIG.signals.vwap_extension_threshold}x during window")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='KOSS')
    parser.add_argument('--date', default='2021-01-27')
    args = parser.parse_args()
    
    run_fixed_backtest(args.symbol, args.date)
