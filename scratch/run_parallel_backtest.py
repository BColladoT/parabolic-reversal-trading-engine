"""
Parallel Backtest - 3,571 Symbols with Multiprocessing

Divides symbols across CPU cores for massive speedup.
"""

import sys
import json
import time
import multiprocessing as mp
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
import pandas as pd
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.backtest.historical_tick_fetcher import tick_fetcher
from src.backtest.extended_universe import EXTENDED_MICRO_CAP_SYMBOLS
from src.risk.ml_simple import InstitutionalRiskManager
from src.strategies import get_strategy


# Configuration
ALL_SYMBOLS = EXTENDED_MICRO_CAP_SYMBOLS
NUM_WORKERS = mp.cpu_count()  # Use all CPU cores
BATCH_SIZE = max(1, len(ALL_SYMBOLS) // NUM_WORKERS)

print(f"Parallel Backtest: {len(ALL_SYMBOLS)} symbols, {NUM_WORKERS} workers, ~{BATCH_SIZE} symbols/worker")


@dataclass
class TradeRecord:
    """Trade record for parallel processing."""
    symbol: str
    date: str
    day_gain_pct: float
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    win: int
    strategy: str
    ml_blocked: bool = False
    risk_score: float = 0.0
    win_probability: float = 0.0
    kelly_fraction: float = 1.0
    recommendation: str = ""
    var_95: float = 0.0
    cvar_95: float = 0.0
    minutes_to_peak: float = 0.0
    vwap_deviation: float = 0.0
    volume_concentration: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


def process_symbol_batch(args):
    """Worker function - processes a batch of symbols."""
    batch_num, symbols, start_date_str, end_date_str, output_dir = args
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    
    # Initialize components per worker
    ml_risk = InstitutionalRiskManager()
    v5_strategy = get_strategy('v5_relaxed_scanner')
    
    trades = []
    stats = {
        'worker_id': batch_num,
        'symbols_processed': 0,
        'setups_found': 0,
        'v5_trades': 0,
        'v5_pnl': 0.0,
        'ml_trades': 0,
        'ml_blocked': 0,
        'ml_pnl': 0.0,
    }
    
    print(f"[Worker {batch_num}] Starting {len(symbols)} symbols...")
    
    for idx, symbol in enumerate(symbols):
        stats['symbols_processed'] += 1
        
        if idx % 10 == 0:
            print(f"[Worker {batch_num}] {idx}/{len(symbols)} symbols done")
        
        # Process all dates for this symbol
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:
                setup = check_setup(symbol, current_date)
                
                if setup:
                    stats['setups_found'] += 1
                    process_setup(setup, v5_strategy, ml_risk, trades, stats)
            
            current_date += timedelta(days=1)
    
    # Save worker results
    worker_file = output_dir / f'worker_{batch_num}_trades.csv'
    if trades:
        df = pd.DataFrame([t.to_dict() for t in trades])
        df.to_csv(worker_file, index=False)
    
    with open(output_dir / f'worker_{batch_num}_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"[Worker {batch_num}] COMPLETE: {stats['setups_found']} setups, {stats['v5_trades']} V5 trades, ${stats['v5_pnl']:,.0f}")
    
    return stats


def check_setup(symbol: str, date: datetime) -> Optional[Dict]:
    """Check if symbol/date is a parabolic setup."""
    try:
        tick_df = tick_fetcher.fetch_combined_tick_data(symbol, date, use_quotes=False)
        if tick_df.is_empty():
            return None
        
        bar_df = tick_fetcher.aggregate_trades_to_bars(tick_df, interval_seconds=60)
        if bar_df.is_empty() or len(bar_df) < 30:
            return None
        
        bars = bar_df.to_pandas()
        bars['timestamp'] = pd.to_datetime(bars['timestamp'])
        bars = bars.sort_values('timestamp')
        
        day_open = bars.iloc[0]['open']
        day_high = bars['high'].max()
        day_gain = (day_high - day_open) / day_open * 100
        
        if day_gain < 30:
            return None
        
        bars['typical'] = (bars['high'] + bars['low'] + bars['close']) / 3
        bars['tp_v'] = bars['typical'] * bars['volume']
        bars['cum_tp_v'] = bars['tp_v'].cumsum()
        bars['cum_vol'] = bars['volume'].cumsum()
        bars['vwap'] = bars['cum_tp_v'] / bars['cum_vol']
        
        return {
            'symbol': symbol,
            'date': date,
            'bars': bars,
            'day_open': day_open,
            'day_high': day_high,
            'day_low': bars['low'].min(),
            'day_gain': day_gain
        }
    except Exception as e:
        return None


def process_setup(setup: Dict, v5_strategy, ml_risk, trades: List, stats: Dict):
    """Process a found setup with both strategies."""
    symbol = setup['symbol']
    date = setup['date']
    date_str = date.strftime('%Y-%m-%d')
    bars = setup['bars']
    
    features = extract_features(bars)
    
    # V5 RELAXED
    try:
        v5_result = v5_strategy.run_tick_backtest(symbol, date, verbose=False)
        if v5_result.total_trades > 0:
            stats['v5_trades'] += 1
            stats['v5_pnl'] += v5_result.total_pnl
            
            trades.append(TradeRecord(
                symbol=symbol,
                date=date_str,
                day_gain_pct=setup['day_gain'],
                entry_price=setup['day_open'],
                exit_price=setup['day_high'],
                shares=0,
                pnl=v5_result.total_pnl,
                win=1 if v5_result.total_pnl > 0 else 0,
                strategy='v5_relaxed',
                ml_blocked=False,
                **features
            ))
    except:
        pass
    
    # V5 INSTITUTIONAL ML
    try:
        raw_data = {'symbol': symbol, 'date': date_str, 'bars': bars.to_dict('records')}
        assessment = ml_risk.assess_trade(raw_data)
        
        if assessment['recommendation'] == 'AVOID':
            stats['ml_blocked'] += 1
            trades.append(TradeRecord(
                symbol=symbol,
                date=date_str,
                day_gain_pct=setup['day_gain'],
                entry_price=0,
                exit_price=0,
                shares=0,
                pnl=0,
                win=0,
                strategy='v5_institutional',
                ml_blocked=True,
                risk_score=assessment['risk_score'],
                win_probability=assessment['win_probability'],
                kelly_fraction=0,
                recommendation='AVOID',
                var_95=assessment['var_95'],
                cvar_95=assessment['cvar_95'],
                **features
            ))
        else:
            kelly = assessment['kelly_fraction']
            entry_bar = bars[(bars['timestamp'].dt.hour >= 10) & (bars['timestamp'].dt.hour < 11)]
            entry_price = entry_bar.iloc[0]['close'] if not entry_bar.empty else bars.iloc[30]['close']
            vwap = bars['vwap'].iloc[-1]
            exit_price = vwap
            
            position_value = 25000 * kelly
            shares = int(position_value / entry_price)
            pnl = (entry_price - exit_price) * shares
            
            stats['ml_trades'] += 1
            stats['ml_pnl'] += pnl
            
            trades.append(TradeRecord(
                symbol=symbol,
                date=date_str,
                day_gain_pct=setup['day_gain'],
                entry_price=entry_price,
                exit_price=exit_price,
                shares=shares,
                pnl=pnl,
                win=1 if pnl > 0 else 0,
                strategy='v5_institutional',
                ml_blocked=False,
                risk_score=assessment['risk_score'],
                win_probability=assessment['win_probability'],
                kelly_fraction=kelly,
                recommendation=assessment['recommendation'],
                var_95=assessment['var_95'],
                cvar_95=assessment['cvar_95'],
                **features
            ))
    except:
        pass


def extract_features(bars: pd.DataFrame) -> Dict:
    """Extract key features from bars."""
    try:
        peak_idx = bars['high'].idxmax()
        peak_time = bars.loc[peak_idx, 'timestamp']
        market_open = bars.iloc[0]['timestamp']
        minutes_to_peak = (peak_time - market_open).total_seconds() / 60
        
        peak_price = bars.loc[peak_idx, 'high']
        vwap_at_peak = bars.loc[peak_idx, 'vwap']
        vwap_deviation = ((peak_price - vwap_at_peak) / vwap_at_peak) * 100
        
        first_hour_mask = bars['timestamp'].dt.hour < 11
        first_hour_vol = bars[first_hour_mask]['volume'].sum() if first_hour_mask.any() else bars.head(30)['volume'].sum()
        total_vol = bars['volume'].sum()
        volume_conc = first_hour_vol / total_vol if total_vol > 0 else 0
        
        return {'minutes_to_peak': minutes_to_peak, 'vwap_deviation': vwap_deviation, 'volume_concentration': volume_conc}
    except:
        return {'minutes_to_peak': 0, 'vwap_deviation': 0, 'volume_concentration': 0}


def combine_results(output_dir: Path, num_workers: int):
    """Combine results from all workers."""
    print("\n[COMBINING] Aggregating results from all workers...")
    
    all_trades = []
    all_stats = {
        'symbols_processed': 0,
        'setups_found': 0,
        'v5_trades': 0,
        'v5_pnl': 0.0,
        'ml_trades': 0,
        'ml_blocked': 0,
        'ml_pnl': 0.0,
    }
    
    for i in range(num_workers):
        stats_file = output_dir / f'worker_{i}_stats.json'
        trades_file = output_dir / f'worker_{i}_trades.csv'
        
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                all_stats['symbols_processed'] += stats['symbols_processed']
                all_stats['setups_found'] += stats['setups_found']
                all_stats['v5_trades'] += stats['v5_trades']
                all_stats['v5_pnl'] += stats['v5_pnl']
                all_stats['ml_trades'] += stats['ml_trades']
                all_stats['ml_blocked'] += stats['ml_blocked']
                all_stats['ml_pnl'] += stats['ml_pnl']
        
        if trades_file.exists():
            df = pd.read_csv(trades_file)
            all_trades.append(df)
    
    # Save combined results
    if all_trades:
        combined_df = pd.concat(all_trades, ignore_index=True)
        combined_df.to_csv(output_dir / 'combined_trades.csv', index=False)
        print(f"  Combined {len(combined_df)} trade records")
    
    with open(output_dir / 'combined_stats.json', 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    return all_stats, all_trades


def main():
    print("="*80)
    print("PARALLEL BACKTEST - 3,571 SYMBOLS")
    print("="*80)
    print(f"Workers: {NUM_WORKERS}")
    print(f"Symbols: {len(ALL_SYMBOLS)}")
    print(f"Period: 2019-01-01 to 2024-12-31")
    print(f"Estimated Speedup: {NUM_WORKERS}x faster")
    print("="*80)
    
    # Setup
    output_dir = Path('reports/parallel_backtest')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    # Split symbols into batches
    symbol_batches = []
    for i in range(NUM_WORKERS):
        start_idx = i * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE if i < NUM_WORKERS - 1 else len(ALL_SYMBOLS)
        batch = ALL_SYMBOLS[start_idx:end_idx]
        symbol_batches.append((i, batch, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), output_dir))
    
    print(f"\nSplit into {NUM_WORKERS} batches:")
    for i, batch in enumerate(symbol_batches):
        print(f"  Worker {i}: {len(batch[1])} symbols")
    
    print(f"\nStarting parallel processing...")
    start_time = time.time()
    
    # Run parallel processing
    with mp.Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(process_symbol_batch, symbol_batches)
    
    elapsed = time.time() - start_time
    
    # Combine results
    final_stats, final_trades = combine_results(output_dir, NUM_WORKERS)
    
    # Print summary
    print("\n" + "="*80)
    print("PARALLEL BACKTEST COMPLETE")
    print("="*80)
    print(f"\nRuntime: {elapsed/3600:.2f} hours")
    print(f"Symbols: {final_stats['symbols_processed']}")
    print(f"Setups: {final_stats['setups_found']}")
    print(f"\nV5 Relaxed: {final_stats['v5_trades']} trades, ${final_stats['v5_pnl']:,.0f}")
    print(f"V5 Institutional: {final_stats['ml_trades']} taken, {final_stats['ml_blocked']} blocked, ${final_stats['ml_pnl']:,.0f}")
    print(f"\nResults saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    # Required for Windows multiprocessing
    mp.set_start_method('spawn', force=True)
    main()
