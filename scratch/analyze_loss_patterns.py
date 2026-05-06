"""
Deep analysis of losing trade patterns for ML feature engineering.
Analyzes tick-level data to identify characteristics of losing trades.
"""
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import pickle

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.backtest.historical_tick_fetcher import tick_fetcher
from src.strategies import get_strategy

# Load losing trades
losing_df = pd.read_csv('reports/losing_trades_analysis.csv')
winning_df = pd.read_csv('reports/full_3527_backtest_results.csv')
winning_df = winning_df[winning_df['pnl'] > 0]

print("="*80)
print("DEEP DIVE: TICK-LEVEL ANALYSIS OF LOSING TRADES")
print("="*80)

# Sample the worst losses for detailed analysis
worst_losses = losing_df.nsmallest(10, 'pnl')

features_list = []

for idx, row in worst_losses.iterrows():
    symbol = row['symbol']
    date_str = row['date']
    pnl = row['pnl']
    date = datetime.strptime(date_str, "%Y-%m-%d")
    
    print(f"\n[{idx+1}/10] Analyzing {symbol} on {date_str} (Loss: ${pnl:,.2f})")
    
    try:
        # Fetch tick data
        tick_df = tick_fetcher.fetch_combined_tick_data(symbol, date, use_quotes=False)
        if tick_df.is_empty():
            print(f"  [SKIP] No tick data")
            continue
        
        # Aggregate to 1-minute bars
        bar_df = tick_fetcher.aggregate_trades_to_bars(tick_df, interval_seconds=60)
        if bar_df.is_empty():
            print(f"  [SKIP] No bar data")
            continue
        
        bars = bar_df.to_pandas()
        bars['timestamp'] = pd.to_datetime(bars['timestamp'])
        bars = bars.sort_values('timestamp')
        
        # Calculate features
        day_open = bars.iloc[0]['open']
        day_high = bars['high'].max()
        day_low = bars['low'].min()
        day_close = bars.iloc[-1]['close']
        
        # VWAP calculation
        bars['typical'] = (bars['high'] + bars['low'] + bars['close']) / 3
        bars['tp_v'] = bars['typical'] * bars['volume']
        bars['cum_tp_v'] = bars['tp_v'].cumsum()
        bars['cum_vol'] = bars['volume'].cumsum()
        bars['vwap'] = bars['cum_tp_v'] / bars['cum_vol']
        
        # Key metrics
        max_gain_pct = (day_high - day_open) / day_open * 100
        max_drawdown_from_high = (day_low - day_high) / day_high * 100
        
        # Volatility metrics
        bars['range'] = bars['high'] - bars['low']
        bars['range_pct'] = bars['range'] / bars['open'] * 100
        avg_range = bars['range_pct'].mean()
        max_range = bars['range_pct'].max()
        
        # Volume patterns
        total_volume = bars['volume'].sum()
        first_hour_volume = bars[bars['timestamp'].dt.hour < 11]['volume'].sum()
        volume_concentration = first_hour_volume / total_volume if total_volume > 0 else 0
        
        # Price action patterns
        bars['body'] = abs(bars['close'] - bars['open'])
        bars['body_pct'] = bars['body'] / bars['open'] * 100
        avg_body = bars['body_pct'].mean()
        
        # VWAP deviation
        max_vwap_deviation = ((bars['high'] - bars['vwap']) / bars['vwap']).max() * 100
        
        # Time to peak
        peak_idx = bars['high'].idxmax()
        peak_time = bars.loc[peak_idx, 'timestamp']
        market_open = bars.iloc[0]['timestamp']
        minutes_to_peak = (peak_time - market_open).total_seconds() / 60
        
        # Recovery from peak
        peak_price = bars.loc[peak_idx, 'high']
        post_peak_bars = bars[bars['timestamp'] > peak_time]
        if not post_peak_bars.empty:
            recovery_low = post_peak_bars['low'].min()
            recovery_pct = (recovery_low - peak_price) / peak_price * 100
        else:
            recovery_pct = 0
        
        print(f"  Max Gain: {max_gain_pct:.1f}% | To Peak: {minutes_to_peak:.0f}min")
        print(f"  VWAP Dev: {max_vwap_deviation:.1f}% | Recovery: {recovery_pct:.1f}%")
        print(f"  Volume Conc: {volume_concentration:.1%} | Avg Range: {avg_range:.2f}%")
        
        features = {
            'symbol': symbol,
            'date': date_str,
            'pnl': pnl,
            'max_gain_pct': max_gain_pct,
            'max_drawdown_from_high': max_drawdown_from_high,
            'avg_range': avg_range,
            'max_range': max_range,
            'volume_concentration': volume_concentration,
            'total_volume': total_volume,
            'avg_body': avg_body,
            'max_vwap_deviation': max_vwap_deviation,
            'minutes_to_peak': minutes_to_peak,
            'recovery_pct': recovery_pct,
            'result': 'loss'
        }
        features_list.append(features)
        
    except Exception as e:
        print(f"  [ERROR] {e}")
        continue

# Now analyze some winning trades for comparison
print("\n" + "="*80)
print("COMPARISON: WINNING TRADE CHARACTERISTICS")
print("="*80)

best_wins = winning_df.nlargest(10, 'pnl')

for idx, row in best_wins.iterrows():
    symbol = row['symbol']
    date_str = row['date']
    pnl = row['pnl']
    date = datetime.strptime(date_str, "%Y-%m-%d")
    
    print(f"\n[{idx+1}/10] Analyzing {symbol} on {date_str} (Win: ${pnl:,.2f})")
    
    try:
        tick_df = tick_fetcher.fetch_combined_tick_data(symbol, date, use_quotes=False)
        if tick_df.is_empty():
            continue
        
        bar_df = tick_fetcher.aggregate_trades_to_bars(tick_df, interval_seconds=60)
        if bar_df.is_empty():
            continue
        
        bars = bar_df.to_pandas()
        bars['timestamp'] = pd.to_datetime(bars['timestamp'])
        bars = bars.sort_values('timestamp')
        
        day_open = bars.iloc[0]['open']
        day_high = bars['high'].max()
        
        bars['typical'] = (bars['high'] + bars['low'] + bars['close']) / 3
        bars['tp_v'] = bars['typical'] * bars['volume']
        bars['cum_tp_v'] = bars['tp_v'].cumsum()
        bars['cum_vol'] = bars['volume'].cumsum()
        bars['vwap'] = bars['cum_tp_v'] / bars['cum_vol']
        
        max_gain_pct = (day_high - day_open) / day_open * 100
        bars['range'] = bars['high'] - bars['low']
        bars['range_pct'] = bars['range'] / bars['open'] * 100
        avg_range = bars['range_pct'].mean()
        
        total_volume = bars['volume'].sum()
        first_hour_volume = bars[bars['timestamp'].dt.hour < 11]['volume'].sum()
        volume_concentration = first_hour_volume / total_volume if total_volume > 0 else 0
        
        max_vwap_deviation = ((bars['high'] - bars['vwap']) / bars['vwap']).max() * 100
        
        peak_idx = bars['high'].idxmax()
        peak_time = bars.loc[peak_idx, 'timestamp']
        market_open = bars.iloc[0]['timestamp']
        minutes_to_peak = (peak_time - market_open).total_seconds() / 60
        
        print(f"  Max Gain: {max_gain_pct:.1f}% | To Peak: {minutes_to_peak:.0f}min")
        print(f"  VWAP Dev: {max_vwap_deviation:.1f}% | Vol Conc: {volume_concentration:.1%}")
        
        features = {
            'symbol': symbol,
            'date': date_str,
            'pnl': pnl,
            'max_gain_pct': max_gain_pct,
            'avg_range': avg_range,
            'volume_concentration': volume_concentration,
            'total_volume': total_volume,
            'max_vwap_deviation': max_vwap_deviation,
            'minutes_to_peak': minutes_to_peak,
            'result': 'win'
        }
        features_list.append(features)
        
    except Exception as e:
        print(f"  [ERROR] {e}")
        continue

# Save features for ML training
features_df = pd.DataFrame(features_list)
features_df.to_csv('reports/ml_features_dataset.csv', index=False)
print("\n" + "="*80)
print(f"[SAVED] Feature dataset: {len(features_df)} samples")
print("Location: reports/ml_features_dataset.csv")
print("="*80)

# Print summary comparison
if len(features_df) > 0:
    print("\nFEATURE COMPARISON: Losses vs Wins")
    print("-"*80)
    
    loss_features = features_df[features_df['result'] == 'loss']
    win_features = features_df[features_df['result'] == 'win']
    
    for col in ['max_gain_pct', 'max_vwap_deviation', 'minutes_to_peak', 
                'volume_concentration', 'avg_range']:
        if col in loss_features.columns and col in win_features.columns:
            loss_avg = loss_features[col].mean()
            win_avg = win_features[col].mean()
            print(f"{col:25s}: Losses={loss_avg:8.2f} | Wins={win_avg:8.2f}")
