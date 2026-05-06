"""Debug entry conditions for a specific setup."""
from datetime import datetime
import pytz
from src.backtest.historical_tick_fetcher import tick_fetcher
from src.backtest.tick_backtest_engine import TickBacktestEngine

def debug_entry(symbol, date_str):
    date = datetime.strptime(date_str, '%Y-%m-%d')
    
    print(f"\n{'='*70}")
    print(f"DEBUG ENTRY: {symbol} on {date_str}")
    print(f"{'='*70}\n")
    
    # Fetch data
    tick_df = tick_fetcher.fetch_combined_tick_data(symbol, date, use_quotes=False)
    
    if tick_df.is_empty():
        print("No tick data!")
        return
    
    print(f"Total ticks: {len(tick_df):,}")
    
    # Aggregate
    bar_df = tick_fetcher.aggregate_trades_to_bars(tick_df, interval_seconds=60)
    print(f"1-minute bars: {len(bar_df)}")
    
    # Get day stats
    day_high = bar_df['high'].max()
    day_open = bar_df['open'].first()
    day_gain = (day_high - day_open) / day_open * 100 if day_open > 0 else 0
    
    print(f"\nDay stats:")
    print(f"  Open: ${day_open:.2f}")
    print(f"  High: ${day_high:.2f}")
    print(f"  Gain: {day_gain:.1f}%")
    
    # Filter to entry window (9:45 - 2:30 PM ET = 13:45 - 18:30 UTC)
    et_tz = pytz.timezone('America/New_York')
    
    entry_bars = []
    for row in bar_df.to_dicts():
        ts = row['timestamp']
        if ts.tzinfo is not None:
            ts_et = ts.astimezone(et_tz)
        else:
            ts_et = ts
        
        if datetime.time(9, 45) <= ts_et.time() <= datetime.time(14, 30):
            entry_bars.append(row)
    
    print(f"\nBars in entry window (9:45 AM - 2:30 PM ET): {len(entry_bars)}")
    
    if len(entry_bars) == 0:
        print("No bars in entry window!")
        return
    
    # Check volume exhaustion
    if len(entry_bars) > 5:
        # First 30 min volume peak
        peak_volume = max(b['volume'] for b in entry_bars[:30])
        print(f"\nVolume peak (first 30 min): {peak_volume:,.0f}")
        print(f"60% threshold: {peak_volume * 0.60:,.0f}")
        
        # Check each bar for entry conditions
        print(f"\nChecking for entry signals (VWAP ext > 1.20, Vol < 60%, Price > 95% of HOD):")
        
        for i, bar in enumerate(entry_bars[30:], 30):  # Start after first 30 min
            volume_ratio = bar['volume'] / peak_volume
            price_proximity = bar['close'] / day_high
            
            # Simplified VWAP check (would need actual VWAP calc)
            if volume_ratio < 0.60 and price_proximity >= 0.95:
                print(f"\n  *** POTENTIAL ENTRY at bar {i} ***")
                print(f"      Time: {bar['timestamp']}")
                print(f"      Price: ${bar['close']:.2f} ({price_proximity*100:.1f}% of HOD)")
                print(f"      Volume: {bar['volume']:,.0f} ({volume_ratio*100:.1f}% of peak)")
                break
        else:
            print("  No entry signal found in this setup")

if __name__ == "__main__":
    debug_entry('AMC', '2021-06-02')
    debug_entry('WWR', '2020-10-05')
