"""Debug V2 engine to see why no entries."""
from datetime import datetime
import pytz
from src.backtest.historical_tick_fetcher import tick_fetcher
from src.backtest.tick_backtest_engine_v2 import TickBacktestEngineV2

et_tz = pytz.timezone('America/New_York')

def debug_setup(symbol, date_str):
    date = datetime.strptime(date_str, '%Y-%m-%d')
    
    print(f"\n{'='*70}")
    print(f"DEBUGGING: {symbol} on {date_str}")
    print(f"{'='*70}\n")
    
    tick_df = tick_fetcher.fetch_combined_tick_data(symbol, date, use_quotes=False)
    bar_df = tick_fetcher.aggregate_trades_to_bars(tick_df, interval_seconds=60)
    
    # Calculate VWAP
    cumulative_tp_v = 0.0
    cumulative_vol = 0.0
    
    bars = []
    for row in bar_df.to_dicts():
        ts = row['timestamp']
        ts_et = ts.astimezone(et_tz) if ts.tzinfo else ts
        
        typical_price = (row['high'] + row['low'] + row['close']) / 3
        cumulative_tp_v += typical_price * row['volume']
        cumulative_vol += row['volume']
        vwap = cumulative_tp_v / cumulative_vol if cumulative_vol > 0 else row['close']
        
        bars.append({
            'time_et': ts_et,
            'close': row['close'],
            'volume': row['volume'],
            'vwap': vwap,
            'vwap_ext': row['close'] / vwap if vwap > 0 else 1.0
        })
    
    day_high = max(b['close'] for b in bars)
    print(f"Day High: ${day_high:.2f}")
    
    # Check entry window
    from datetime import time as dt_time
    print(f"\nChecking 9:45 AM - 2:30 PM ET (VWAP > 1.15x, Vol < 70% recent, Price > 93% HOD):\n")
    
    candidates = []
    volume_window = []
    
    for i, b in enumerate(bars):
        t = b['time_et']
        if not isinstance(t, datetime):
            continue
        
        # Entry window
        if not (dt_time(9, 45) <= t.time() <= dt_time(14, 30)):
            continue
        
        # Update volume window
        volume_window.append(b['volume'])
        if len(volume_window) > 10:
            volume_window.pop(0)
        
        recent_peak = max(volume_window) if volume_window else b['volume']
        vol_ratio = b['volume'] / recent_peak if recent_peak > 0 else 1.0
        prox = b['close'] / day_high if day_high > 0 else 0
        
        vwap_ok = b['vwap_ext'] >= 1.15
        vol_ok = vol_ratio <= 0.70
        prox_ok = prox >= 0.93
        
        met = []
        if vwap_ok:
            met.append("VWAP")
        if vol_ok:
            met.append("VOL")
        if prox_ok:
            met.append("PROX")
        
        if len(met) >= 2:  # At least 2 criteria
            candidates.append({
                'time': t.strftime('%H:%M'),
                'price': b['close'],
                'vwap_ext': b['vwap_ext'],
                'vol_ratio': vol_ratio,
                'prox': prox,
                'met': met,
                'all': len(met) == 3
            })
            
            if len(candidates) <= 10:  # Show first 10
                status = "ENTRY!" if len(met) == 3 else "partial"
                print(f"{t.strftime('%H:%M')} | ${b['close']:.2f} | VWAP:{b['vwap_ext']:.2f}x | "
                      f"Vol:{vol_ratio:.2f} | Prox:{prox:.2f} | {','.join(met)} | {status}")
    
    print(f"\nTotal candidates (2+ criteria): {len(candidates)}")
    print(f"Full entries (3 criteria): {len([c for c in candidates if c['all']])}")
    
    # Show why no full entries
    if len([c for c in candidates if c['all']]) == 0 and candidates:
        print("\nPartial matches (missing 1 criterion):")
        for c in candidates[:5]:
            missing = []
            if c['vwap_ext'] < 1.15:
                missing.append(f"VWAP ({c['vwap_ext']:.2f}x)")
            if c['vol_ratio'] > 0.70:
                missing.append(f"VOL ({c['vol_ratio']:.2f})")
            if c['prox'] < 0.93:
                missing.append(f"PROX ({c['prox']:.2f})")
            print(f"  {c['time']}: missing {', '.join(missing)}")

if __name__ == "__main__":
    debug_setup('WWR', '2020-10-05')
