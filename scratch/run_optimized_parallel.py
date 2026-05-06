"""
Optimized Parallel Backtest with Cache-First Strategy

Phase 1: Download missing data sequentially (respect API limits)
Phase 2: Process all cached data in parallel (CPU-bound, no API)
"""

import sys
import json
import time
import multiprocessing as mp
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
import pandas as pd
import polars as pl

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.backtest.historical_tick_fetcher import tick_fetcher
from src.backtest.extended_universe import EXTENDED_MICRO_CAP_SYMBOLS


ALL_SYMBOLS = EXTENDED_MICRO_CAP_SYMBOLS
CACHE_DIR = Path("data/cache/ticks")
NUM_WORKERS = mp.cpu_count()

print(f"Optimized Parallel Backtest: {len(ALL_SYMBOLS)} symbols, {NUM_WORKERS} workers")


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


def check_cache_status(symbols: List[str], start_date: datetime, end_date: datetime) -> Tuple[Set[str], Set[str]]:
    """
    Check which symbol/date combinations are cached vs need downloading.
    Returns: (cached_combinations, missing_combinations)
    Each combination is "SYMBOL_YYYYMMDD"
    """
    cached = set()
    missing = set()
    
    current_date = start_date
    trading_days = 0
    
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Trading day
            date_str = current_date.strftime("%Y%m%d")
            trading_days += 1
            
            for symbol in symbols:
                cache_key = f"{symbol}_{date_str}"
                cache_path = CACHE_DIR / f"{symbol}_trades_{date_str}.parquet"
                
                if cache_path.exists():
                    cached.add(cache_key)
                else:
                    missing.add(cache_key)
        
        current_date += timedelta(days=1)
    
    return cached, missing, trading_days


def download_missing_data(missing_combinations: Set[str], max_per_minute: int = 180):
    """
    Download missing data sequentially with rate limiting.
    Respects Alpaca's ~200 req/min limit.
    """
    if not missing_combinations:
        print("All data already cached!")
        return
    
    print(f"\n[PHASE 1] Downloading {len(missing_combinations)} missing datasets...")
    print(f"Rate limit: {max_per_minute} requests/minute")
    
    total = len(missing_combinations)
    downloaded = 0
    start_time = time.time()
    
    # Parse missing combinations
    to_download = []
    for combo in missing_combinations:
        parts = combo.split('_')
        symbol = '_'.join(parts[:-1])  # Handle symbols with underscores
        date_str = parts[-1]
        date = datetime.strptime(date_str, "%Y%m%d")
        to_download.append((symbol, date))
    
    # Download with rate limiting
    for symbol, date in to_download:
        try:
            # Rate limiting
            elapsed = time.time() - start_time
            expected_time = (downloaded / max_per_minute) * 60
            if elapsed < expected_time:
                sleep_time = expected_time - elapsed
                time.sleep(sleep_time)
            
            # Download
            market_open = date.replace(hour=9, minute=30)
            market_close = date.replace(hour=16, minute=0)
            
            tick_fetcher.fetch_historical_trades(symbol, market_open, market_close, use_cache=True)
            
            downloaded += 1
            if downloaded % 10 == 0:
                pct = (downloaded / total) * 100
                eta_mins = (total - downloaded) / max_per_minute
                print(f"  Downloaded {downloaded}/{total} ({pct:.1f}%) - ETA: {eta_mins:.0f} min")
                
        except Exception as e:
            print(f"  Error downloading {symbol} {date}: {e}")
    
    print(f"[PHASE 1 COMPLETE] Downloaded {downloaded} datasets")


def process_symbol_batch(args):
    """
    Worker function - processes symbols using ONLY cached data (no API calls).
    This runs in parallel across all CPU cores.
    """
    batch_num, symbols, start_date_str, end_date_str, output_dir = args
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    
    # Import here to avoid pickling issues
    from src.risk.ml_simple import InstitutionalRiskManager
    from src.strategies import get_strategy
    
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
    
    print(f"[Worker {batch_num}] Processing {len(symbols)} symbols from cache...")
    
    for idx, symbol in enumerate(symbols):
        stats['symbols_processed'] += 1
        
        if idx % 10 == 0:
            print(f"[Worker {batch_num}] {idx}/{len(symbols)} done, {stats['setups_found']} setups")
        
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:
                setup = check_setup_cached(symbol, current_date)
                if setup:
                    stats['setups_found'] += 1
                    process_setup(setup, v5_strategy, ml_risk, trades, stats)
            current_date += timedelta(days=1)
    
    # Save results
    worker_dir = output_dir / f'worker_{batch_num}'
    worker_dir.mkdir(exist_ok=True, parents=True)
    
    if trades:
        df = pd.DataFrame([t.to_dict() for t in trades])
        df.to_csv(worker_dir / 'trades.csv', index=False)
    
    with open(worker_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"[Worker {batch_num}] DONE: {stats['setups_found']} setups, ${stats['v5_pnl']:,.0f}")
    return stats


def check_setup_cached(symbol: str, date: datetime) -> Optional[Dict]:
    """Check setup using ONLY cached data (no API calls)."""
    try:
        # This will only read from cache, no API call
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
            'symbol': symbol, 'date': date, 'bars': bars,
            'day_open': day_open, 'day_high': day_high,
            'day_low': bars['low'].min(), 'day_gain': day_gain
        }
    except:
        return None


def process_setup(setup: Dict, v5_strategy, ml_risk, trades: List, stats: Dict):
    """Process a setup with V5 and ML strategies."""
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


def extract_features(bars: pd.DataFrame) -> Dict:
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


def main():
    print("="*80)
    print("OPTIMIZED PARALLEL BACKTEST (Cache-First Strategy)")
    print("="*80)
    
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2024, 12, 31)
    output_dir = Path('reports/parallel_optimized')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Phase 0: Check cache status
    print("\n[PHASE 0] Checking cache status...")
    cached, missing, trading_days = check_cache_status(ALL_SYMBOLS, start_date, end_date)
    
    total_needed = len(ALL_SYMBOLS) * trading_days
    cache_pct = (len(cached) / total_needed) * 100 if total_needed > 0 else 0
    
    print(f"  Total symbol-days needed: {total_needed:,}")
    print(f"  Already cached: {len(cached):,} ({cache_pct:.1f}%)")
    print(f"  Missing: {len(missing):,}")
    print(f"  Trading days: {trading_days}")
    
    # Phase 1: Download missing data (sequential, rate-limited)
    if missing:
        download_missing_data(missing)
    
    # Phase 2: Process in parallel (no API calls, CPU-bound)
    print("\n[PHASE 2] Processing all cached data in parallel...")
    
    # Split symbols into batches
    symbols_per_worker = len(ALL_SYMBOLS) // NUM_WORKERS
    batches = []
    
    for i in range(NUM_WORKERS):
        start_idx = i * symbols_per_worker
        end_idx = start_idx + symbols_per_worker if i < NUM_WORKERS - 1 else len(ALL_SYMBOLS)
        batch_symbols = ALL_SYMBOLS[start_idx:end_idx]
        batches.append((i, batch_symbols, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), output_dir))
    
    print(f"  Split into {NUM_WORKERS} batches of ~{symbols_per_worker} symbols each")
    
    start_time = time.time()
    
    # Run parallel processing
    with mp.Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(process_symbol_batch, batches)
    
    elapsed = time.time() - start_time
    
    # Combine results
    print("\n[PHASE 3] Combining results...")
    
    all_trades = []
    combined_stats = {
        'symbols_processed': 0,
        'setups_found': 0,
        'v5_trades': 0,
        'v5_pnl': 0.0,
        'ml_trades': 0,
        'ml_blocked': 0,
        'ml_pnl': 0.0,
    }
    
    for i in range(NUM_WORKERS):
        worker_dir = output_dir / f'worker_{i}'
        stats_file = worker_dir / 'stats.json'
        trades_file = worker_dir / 'trades.csv'
        
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                for key in combined_stats:
                    combined_stats[key] += stats.get(key, 0)
        
        if trades_file.exists():
            df = pd.read_csv(trades_file)
            all_trades.append(df)
    
    if all_trades:
        combined_df = pd.concat(all_trades, ignore_index=True)
        combined_df.to_csv(output_dir / 'combined_trades.csv', index=False)
    
    with open(output_dir / 'combined_stats.json', 'w') as f:
        json.dump(combined_stats, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("BACKTEST COMPLETE")
    print("="*80)
    print(f"\nProcessing Time: {elapsed/3600:.2f} hours")
    print(f"Symbols: {combined_stats['symbols_processed']}")
    print(f"Setups: {combined_stats['setups_found']}")
    print(f"\nV5: {combined_stats['v5_trades']} trades, ${combined_stats['v5_pnl']:,.0f}")
    print(f"ML: {combined_stats['ml_trades']} taken, {combined_stats['ml_blocked']} blocked, ${combined_stats['ml_pnl']:,.0f}")
    print(f"\nResults: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
