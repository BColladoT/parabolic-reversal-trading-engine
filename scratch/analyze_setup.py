"""Analyze a specific setup to understand why no entry."""
from datetime import datetime
import pytz
from src.backtest.historical_tick_fetcher import tick_fetcher

et_tz = pytz.timezone('America/New_York')

def analyze_setup(symbol, date_str):
    date = datetime.strptime(date_str, '%Y-%m-%d')
    
    print(f"\n{'='*70}")
    print(f"ANALYZING: {symbol} on {date_str}")
    print(f"{'='*70}\n")
    
    # Fetch tick data
    tick_df = tick_fetcher.fetch_combined_tick_data(symbol, date, use_quotes=False)
    
    if tick_df.is_empty():
        print("No tick data!")
        return
    
    # Aggregate to bars
    bar_df = tick_fetcher.aggregate_trades_to_bars(tick_df, interval_seconds=60)
    
    print(f"Total 1-minute bars: {len(bar_df)}")
    
    # Calculate VWAP manually
    cumulative_tp_v = 0.0
    cumulative_vol = 0.0
    
    bars_with_metrics = []
    
    for row in bar_df.to_dicts():
        ts = row['timestamp']
        
        # Convert to ET
        if ts.tzinfo is not None:
            ts_et = ts.astimezone(et_tz)
        else:
            ts_et = ts
        
        # Calculate VWAP
        typical_price = (row['high'] + row['low'] + row['close']) / 3
        cumulative_tp_v += typical_price * row['volume']
        cumulative_vol += row['volume']
        vwap = cumulative_tp_v / cumulative_vol if cumulative_vol > 0 else row['close']
        
        # Calculate metrics
        vwap_ext = row['close'] / vwap if vwap > 0 else 1.0
        
        bars_with_metrics.append({
            'timestamp': ts,
            'time_et': ts_et,
            'close': row['close'],
            'volume': row['volume'],
            'vwap': vwap,
            'vwap_ext': vwap_ext,
            'high': row['high']
        })
    
    # Find day high
    day_high = max(b['high'] for b in bars_with_metrics)
    
    # Find volume peak in first 30 minutes
    market_open = None
    volume_peak = 0
    
    for b in bars_with_metrics:
        if market_open is None:
            market_open = b['time_et']
        
        elapsed = (b['time_et'] - market_open).total_seconds() / 60
        if elapsed <= 30:
            if b['volume'] > volume_peak:
                volume_peak = b['volume']
    
    print(f"\nDay High: ${day_high:.2f}")
    print(f"Volume Peak (first 30 min): {volume_peak:,.0f}")
    
    # Check entry window (9:45 - 2:30 PM ET)
    print(f"\nChecking entry window (9:45 AM - 2:30 PM ET):")
    print(f"Criteria: VWAP > 1.20x, Vol < 60% of peak, Price > 95% of HOD\n")
    
    candidates = []
    
    for b in bars_with_metrics:
        t = b['time_et']
        time_only = t.time() if hasattr(t, 'time') else t
        
        # Check if in entry window
        from datetime import time as dt_time
        if not (dt_time(9, 45) <= time_only <= dt_time(14, 30)):
            continue
        
        # Calculate criteria
        vwap_ok = b['vwap_ext'] >= 1.20
        vol_ratio = b['volume'] / volume_peak if volume_peak > 0 else 1.0
        vol_ok = vol_ratio < 0.60
        prox = b['close'] / day_high if day_high > 0 else 0
        prox_ok = prox >= 0.95
        
        status = []
        if vwap_ok:
            status.append("VWAP")
        if vol_ok:
            status.append("VOL")
        if prox_ok:
            status.append("PROX")
        
        if len(status) >= 2:  # At least 2 criteria
            candidates.append({
                'time': t.strftime('%H:%M'),
                'price': b['close'],
                'vwap_ext': b['vwap_ext'],
                'vol_ratio': vol_ratio,
                'prox': prox,
                'all_met': vwap_ok and vol_ok and prox_ok
            })
            
            print(f"  {t.strftime('%H:%M')} | ${b['close']:.2f} | VWAP: {b['vwap_ext']:.2f}x | "
                  f"Vol: {vol_ratio:.2f} | Prox: {prox:.2f} | {' | '.join(status)}")
    
    from datetime import time as dt_time
    print(f"\n{'='*70}")
    print(f"SUMMARY:")
    checked = 0
    for b in bars_with_metrics:
        to = b['time_et'].time() if hasattr(b['time_et'], 'time') else b['time_et']
        if dt_time(9, 45) <= to <= dt_time(14, 30):
            checked += 1
    print(f"  Total bars checked: {checked}")
    print(f"  Bars with 2+ criteria: {len(candidates)}")
    print(f"  Full entry signals: {len([c for c in candidates if c['all_met']])}")
    
    if len([c for c in candidates if c['all_met']]) == 0:
        print(f"\n  No full entry signals found!")
        print(f"  Most common missing criterion: Need to check individual bars")

if __name__ == "__main__":
    # Test several setups
    analyze_setup('WWR', '2020-10-05')
