"""
Parallel Backtest - Process ONLY Cached Symbols

Fastest option: Skip downloading, process existing cache in parallel.
~719 symbols with cached data → ~2-3 hours completion.
"""

import sys
import json
import time
import multiprocessing as mp
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
import pandas as pd

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


@dataclass
class TradeRecord:
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


def get_cached_symbols() -> Set[str]:
    """Get list of symbols with cached tick data."""
    cache_dir = Path("data/cache/ticks")
    files = list(cache_dir.glob("*_trades_*.parquet"))
    
    symbols = set()
    for f in files:
        parts = f.stem.split('_')
        if len(parts) >= 3 and parts[-2] == 'trades':
            symbol = '_'.join(parts[:-2])
            symbols.add(symbol)
    
    return symbols


def process_symbol_batch(args):
    """Worker function - processes symbols from cache only."""
    batch_num, symbols, start_date_str, end_date_str, output_dir = args
    
    # Imports inside worker to avoid pickling issues
    from src.backtest.historical_tick_fetcher import tick_fetcher
    from src.risk.ml_simple import InstitutionalRiskManager
    from src.strategies import get_strategy
    
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    
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
        
        if idx % 5 == 0:
            print(f"[Worker {batch_num}] {idx+1}/{len(symbols)} symbols, {stats['setups_found']} setups, ${stats['v5_pnl']:,.0f}")
            # Save checkpoint
            with open(output_dir / f'worker_{batch_num}_stats.json', 'w') as f:
                json.dump(stats, f, indent=2)
        
        # Process all dates
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:
                try:
                    import polars as pl
                    cache_path = Path(f"data/cache/ticks/{symbol}_trades_{current_date.strftime('%Y%m%d')}.parquet")
                    
                    if not cache_path.exists():
                        current_date += timedelta(days=1)
                        continue
                    
                    tick_df = pl.read_parquet(cache_path)
                    if tick_df.is_empty():
                        current_date += timedelta(days=1)
                        continue
                    
                    # Aggregate to bars
                    from src.backtest.historical_tick_fetcher import tick_fetcher
                    bar_df = tick_fetcher.aggregate_trades_to_bars(tick_df, interval_seconds=60)
                    
                    if bar_df.is_empty() or len(bar_df) < 30:
                        current_date += timedelta(days=1)
                        continue
                    
                    import pandas as pd
                    bars = bar_df.to_pandas()
                    bars['timestamp'] = pd.to_datetime(bars['timestamp'])
                    bars = bars.sort_values('timestamp')
                    
                    day_open = bars.iloc[0]['open']
                    day_high = bars['high'].max()
                    day_gain = (day_high - day_open) / day_open * 100
                    
                    if day_gain >= 30:
                        bars['typical'] = (bars['high'] + bars['low'] + bars['close']) / 3
                        bars['tp_v'] = bars['typical'] * bars['volume']
                        bars['cum_tp_v'] = bars['tp_v'].cumsum()
                        bars['cum_vol'] = bars['volume'].cumsum()
                        bars['vwap'] = bars['cum_tp_v'] / bars['cum_vol']
                        
                        setup = {
                            'symbol': symbol, 'date': current_date, 'bars': bars,
                            'day_open': day_open, 'day_high': day_high,
                            'day_low': bars['low'].min(), 'day_gain': day_gain
                        }
                        
                        stats['setups_found'] += 1
                        process_setup(setup, v5_strategy, ml_risk, trades, stats)
                        
                except Exception as e:
                    pass
            
            current_date += timedelta(days=1)
    
    # Save final results
    if trades:
        df = pd.DataFrame([t.to_dict() for t in trades])
        df.to_csv(output_dir / f'worker_{batch_num}_trades.csv', index=False)
    
    with open(output_dir / f'worker_{batch_num}_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"[Worker {batch_num}] COMPLETE: {stats['setups_found']} setups, ${stats['v5_pnl']:,.0f}")
    return stats


def process_setup(setup, v5_strategy, ml_risk, trades, stats):
    """Process a setup."""
    symbol = setup['symbol']
    date = setup['date']
    date_str = date.strftime('%Y-%m-%d')
    bars = setup['bars']
    
    features = extract_features(bars)
    
    # V5
    try:
        result = v5_strategy.run_tick_backtest(symbol, date, verbose=False)
        if result.total_trades > 0:
            stats['v5_trades'] += 1
            stats['v5_pnl'] += result.total_pnl
            trades.append(TradeRecord(
                symbol=symbol, date=date_str, day_gain_pct=setup['day_gain'],
                entry_price=setup['day_open'], exit_price=setup['day_high'],
                shares=0, pnl=result.total_pnl,
                win=1 if result.total_pnl > 0 else 0,
                strategy='v5_relaxed', ml_blocked=False, **features
            ))
    except:
        pass
    
    # ML
    try:
        raw = {'symbol': symbol, 'date': date_str, 'bars': bars.to_dict('records')}
        assessment = ml_risk.assess_trade(raw)
        
        if assessment['recommendation'] == 'AVOID':
            stats['ml_blocked'] += 1
            trades.append(TradeRecord(
                symbol=symbol, date=date_str, day_gain_pct=setup['day_gain'],
                entry_price=0, exit_price=0, shares=0, pnl=0, win=0,
                strategy='v5_institutional', ml_blocked=True,
                risk_score=assessment['risk_score'],
                win_probability=assessment['win_probability'],
                kelly_fraction=0, recommendation='AVOID',
                var_95=assessment['var_95'], cvar_95=assessment['cvar_95'],
                **features
            ))
        else:
            kelly = assessment['kelly_fraction']
            entry_bar = bars[(bars['timestamp'].dt.hour >= 10) & (bars['timestamp'].dt.hour < 11)]
            entry_price = entry_bar.iloc[0]['close'] if not entry_bar.empty else bars.iloc[30]['close']
            exit_price = bars['vwap'].iloc[-1]
            
            shares = int(25000 * kelly / entry_price)
            pnl = (entry_price - exit_price) * shares
            
            stats['ml_trades'] += 1
            stats['ml_pnl'] += pnl
            
            trades.append(TradeRecord(
                symbol=symbol, date=date_str, day_gain_pct=setup['day_gain'],
                entry_price=entry_price, exit_price=exit_price,
                shares=shares, pnl=pnl, win=1 if pnl > 0 else 0,
                strategy='v5_institutional', ml_blocked=False,
                risk_score=assessment['risk_score'],
                win_probability=assessment['win_probability'],
                kelly_fraction=kelly, recommendation=assessment['recommendation'],
                var_95=assessment['var_95'], cvar_95=assessment['cvar_95'],
                **features
            ))
    except:
        pass


def extract_features(bars):
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


def combine_results(output_dir, num_workers):
    """Combine worker results."""
    print("\n[COMBINING] Aggregating results...")
    
    all_trades = []
    stats = {
        'symbols_processed': 0, 'setups_found': 0,
        'v5_trades': 0, 'v5_pnl': 0.0,
        'ml_trades': 0, 'ml_blocked': 0, 'ml_pnl': 0.0,
    }
    
    for i in range(num_workers):
        stats_file = output_dir / f'worker_{i}_stats.json'
        trades_file = output_dir / f'worker_{i}_trades.csv'
        
        if stats_file.exists():
            with open(stats_file) as f:
                s = json.load(f)
                for k in stats:
                    stats[k] += s.get(k, 0)
        
        if trades_file.exists():
            df = pd.read_csv(trades_file)
            all_trades.append(df)
    
    if all_trades:
        combined = pd.concat(all_trades, ignore_index=True)
        combined.to_csv(output_dir / 'combined_trades.csv', index=False)
        print(f"  Saved {len(combined)} trade records")
    
    with open(output_dir / 'combined_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats


def main():
    print("="*80)
    print("CACHED-ONLY PARALLEL BACKTEST")
    print("="*80)
    
    # Get cached symbols
    cached_symbols = sorted(get_cached_symbols())
    print(f"\nFound {len(cached_symbols)} symbols with cached tick data")
    
    if len(cached_symbols) == 0:
        print("ERROR: No cached data found!")
        return
    
    # Setup
    output_dir = Path('reports/cached_parallel_backtest')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    num_workers = min(mp.cpu_count(), 8)  # Max 8 workers
    symbols_per_worker = len(cached_symbols) // num_workers
    
    print(f"\nWorkers: {num_workers}")
    print(f"Symbols per worker: ~{symbols_per_worker}")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Estimated time: ~2-3 hours")
    print("\n" + "="*80)
    
    # Create batches
    batches = []
    for i in range(num_workers):
        start_idx = i * symbols_per_worker
        end_idx = start_idx + symbols_per_worker if i < num_workers - 1 else len(cached_symbols)
        batch = cached_symbols[start_idx:end_idx]
        batches.append((i, batch, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), output_dir))
    
    # Run parallel
    start_time = time.time()
    
    with mp.Pool(processes=num_workers) as pool:
        pool.map(process_symbol_batch, batches)
    
    elapsed = time.time() - start_time
    
    # Combine
    final_stats = combine_results(output_dir, num_workers)
    
    # Summary
    print("\n" + "="*80)
    print("BACKTEST COMPLETE")
    print("="*80)
    print(f"\nRuntime: {elapsed/3600:.2f} hours")
    print(f"Symbols: {final_stats['symbols_processed']}")
    print(f"Setups: {final_stats['setups_found']}")
    print(f"\nV5: {final_stats['v5_trades']} trades, ${final_stats['v5_pnl']:,.0f}")
    print(f"ML: {final_stats['ml_trades']} taken, {final_stats['ml_blocked']} blocked, ${final_stats['ml_pnl']:,.0f}")
    print(f"\nResults: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
