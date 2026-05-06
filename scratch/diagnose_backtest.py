"""
Diagnose why no trades are being generated in backtest.
"""
from datetime import datetime
import polars as pl
from src.backtest.historical_tick_fetcher import tick_fetcher
from src.indicators.numba_kernels import calculate_vwap_numba

def diagnose_symbol(symbol, date_str):
    date = datetime.strptime(date_str, '%Y-%m-%d')
    
    print(f"\n{'='*60}")
    print(f"DIAGNOSIS: {symbol} on {date_str}")
    print(f"{'='*60}\n")
    
    # Fetch tick data
    tick_df = tick_fetcher.fetch_combined_tick_data(symbol, date, use_quotes=False)
    print(f"Total ticks: {len(tick_df):,}")
    
    if tick_df.is_empty():
        print("No tick data!")
        return
    
    # Aggregate to bars
    bar_df = tick_fetcher.aggregate_trades_to_bars(tick_df, interval_seconds=60)
    print(f"1-minute bars: {len(bar_df)}")
    
    # Calculate VWAP
    highs = bar_df['high'].to_numpy()
    lows = bar_df['low'].to_numpy()
    closes = bar_df['close'].to_numpy()
    volumes = bar_df['volume'].to_numpy()
    vwap_values = calculate_vwap_numba(highs, lows, closes, volumes)
    
    bar_df = bar_df.with_columns([
        pl.Series('vwap', vwap_values),
        (pl.col('close') / pl.col('vwap')).alias('vwap_extension'),
    ])
    
    # Filter to entry window (9:45 AM - 2:30 PM ET)
    entry_window = bar_df.filter(
        (pl.col('timestamp').dt.hour() >= 9) & 
        (pl.col('timestamp').dt.hour() < 14)
    )
    
    if entry_window.is_empty():
        print("No data in entry window!")
        return
    
    # Check morning session specifically (9:45 - 11:00)
    morning = entry_window.filter(
        ((pl.col('timestamp').dt.hour() == 9) & (pl.col('timestamp').dt.minute() >= 45)) |
        (pl.col('timestamp').dt.hour() == 10)
    )
    
    print(f"\nMorning session (9:45-11:00):")
    print(f"  Bars: {len(morning)}")
    
    if len(morning) > 0:
        day_high = bar_df['high'].max()
        print(f"  Day High: ${day_high:.2f}")
        print(f"  Volume peak (first 30 min): {morning['volume'].head(30).max():,.0f}")
        print(f"  VWAP extension range: {morning['vwap_extension'].min():.2f}x - {morning['vwap_extension'].max():.2f}x")
        
        # Check for volume exhaustion
        peak_volume = morning['volume'].head(30).max()
        print(f"\n  Volume exhaustion check (< 60% of peak = {peak_volume * 0.60:,.0f}):")
        
        exhausted = morning.filter(pl.col('volume') < peak_volume * 0.60)
        print(f"  Bars with exhausted volume: {len(exhausted)}")
        
        if len(exhausted) > 0:
            # Check if price near high
            near_high = exhausted.filter((pl.col('close') / day_high) >= 0.95)
            print(f"  Exhausted bars near day high: {len(near_high)}")
            
            # Check VWAP extension
            extended = near_high.filter(pl.col('vwap_extension') >= 1.20)
            print(f"  Extended bars (VWAP >= 1.20x): {len(extended)}")
            
            if len(extended) > 0:
                print(f"\n  POTENTIAL ENTRY FOUND!")
                for row in extended.head(3).to_dicts():
                    print(f"    {row['timestamp']}: ${row['close']:.2f}, VWAP: {row['vwap_extension']:.2f}x, Vol: {row['volume']:,.0f}")

if __name__ == "__main__":
    # Test several known parabolic stocks
    tests = [
        ('AMC', '2021-06-02'),
        ('GME', '2021-01-27'),
        ('WWR', '2020-10-05'),
        ('RENT', '2024-04-11'),
    ]
    
    for symbol, date in tests:
        diagnose_symbol(symbol, date)
