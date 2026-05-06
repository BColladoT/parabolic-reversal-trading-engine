"""
Batch Worker - Process a chunk of symbols independently.

Usage: python run_batch_worker.py <worker_id> <total_workers>
Example: python run_batch_worker.py 0 8  # Worker 0 of 8
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import pandas as pd

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.backtest.historical_tick_fetcher import tick_fetcher
from src.backtest.extended_universe import EXTENDED_MICRO_CAP_SYMBOLS
from src.risk.ml_simple import InstitutionalRiskManager
from src.strategies import get_strategy


ALL_SYMBOLS = EXTENDED_MICRO_CAP_SYMBOLS


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


def main():
    if len(sys.argv) != 3:
        print("Usage: python run_batch_worker.py <worker_id> <total_workers>")
        print("Example: python run_batch_worker.py 0 8")
        sys.exit(1)
    
    worker_id = int(sys.argv[1])
    total_workers = int(sys.argv[2])
    
    # Calculate symbol range for this worker
    symbols_per_worker = len(ALL_SYMBOLS) // total_workers
    start_idx = worker_id * symbols_per_worker
    end_idx = start_idx + symbols_per_worker if worker_id < total_workers - 1 else len(ALL_SYMBOLS)
    my_symbols = ALL_SYMBOLS[start_idx:end_idx]
    
    print("="*80)
    print(f"BATCH WORKER {worker_id}/{total_workers}")
    print("="*80)
    print(f"Symbols: {start_idx} to {end_idx-1} ({len(my_symbols)} symbols)")
    print(f"Date Range: 2019-01-01 to 2024-12-31")
    print("="*80 + "\n")
    
    # Initialize
    output_dir = Path(f'reports/parallel_backtest/worker_{worker_id}')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    ml_risk = InstitutionalRiskManager()
    v5_strategy = get_strategy('v5_relaxed_scanner')
    
    trades = []
    stats = {
        'worker_id': worker_id,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'symbols_total': len(my_symbols),
        'symbols_done': 0,
        'setups_found': 0,
        'v5_trades': 0,
        'v5_pnl': 0.0,
        'ml_trades': 0,
        'ml_blocked': 0,
        'ml_pnl': 0.0,
        'start_time': datetime.now().isoformat(),
    }
    
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2024, 12, 31)
    start_time = time.time()
    
    # Process symbols
    for idx, symbol in enumerate(my_symbols):
        stats['symbols_done'] = idx + 1
        
        if idx % 10 == 0 or idx == len(my_symbols) - 1:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed * 60 if elapsed > 0 else 0
            print(f"[Worker {worker_id}] {idx+1}/{len(my_symbols)} symbols | {stats['setups_found']} setups | ${stats['v5_pnl']:,.0f} P&L | {rate:.1f} sym/min")
            
            # Save checkpoint
            with open(output_dir / 'stats.json', 'w') as f:
                json.dump(stats, f, indent=2)
        
        # Process all dates
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:
                setup = check_setup(symbol, current_date)
                if setup:
                    stats['setups_found'] += 1
                    process_setup(setup, v5_strategy, ml_risk, trades, stats)
            current_date += timedelta(days=1)
    
    # Save final results
    if trades:
        df = pd.DataFrame([t.to_dict() for t in trades])
        df.to_csv(output_dir / 'trades.csv', index=False)
    
    stats['end_time'] = datetime.now().isoformat()
    stats['runtime_seconds'] = time.time() - start_time
    
    with open(output_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n[Worker {worker_id}] COMPLETE!")
    print(f"  Runtime: {stats['runtime_seconds']/3600:.2f} hours")
    print(f"  Setups: {stats['setups_found']}")
    print(f"  V5 Trades: {stats['v5_trades']}, P&L: ${stats['v5_pnl']:,.0f}")
    print(f"  Results: {output_dir}")


def check_setup(symbol: str, date: datetime) -> Optional[Dict]:
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
            'symbol': symbol, 'date': date, 'bars': bars,
            'day_open': day_open, 'day_high': day_high,
            'day_low': bars['low'].min(), 'day_gain': day_gain
        }
    except:
        return None


def process_setup(setup: Dict, v5_strategy, ml_risk, trades: List, stats: Dict):
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


if __name__ == "__main__":
    main()
