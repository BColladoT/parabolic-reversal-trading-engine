"""
Complete Fresh Backtest - Full 6-Year Scan with ML Risk Engine

This script:
1. Scans ALL 3,527 symbols from 2019-2024 (not using cached setups)
2. For each trading day, checks for parabolic moves
3. When a setup is found, runs ML risk assessment in real-time
4. Simulates trade execution with risk-managed position sizing
5. Tracks portfolio, P&L, and statistics in real-time
6. Generates comprehensive report at completion

Usage:
    python run_complete_fresh_backtest.py
    
Estimated Runtime: 6-8 hours for full 3,527 symbols × 6 years
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
import pandas as pd
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.backtest.historical_tick_fetcher import tick_fetcher
from src.backtest.historical_screener import HistoricalParabolicScreener
from src.risk.ml_simple import InstitutionalRiskManager
from src.strategies import get_strategy


@dataclass
class TradeRecord:
    """Complete trade record with risk assessment."""
    symbol: str
    date: str
    day_gain_pct: float
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    win: int
    strategy: str
    
    # ML Risk Assessment
    ml_blocked: bool = False
    risk_score: float = 0.0
    win_probability: float = 0.0
    kelly_fraction: float = 1.0
    recommendation: str = ""
    var_95: float = 0.0
    cvar_95: float = 0.0
    
    # Market features
    minutes_to_peak: float = 0.0
    vwap_deviation: float = 0.0
    volume_concentration: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


class FreshBacktestEngine:
    """
    Complete fresh backtest engine that scans everything.
    """
    
    def __init__(self, start_date: datetime, end_date: datetime):
        self.start_date = start_date
        self.end_date = end_date
        
        # Load all symbols
        self.symbols = self._load_all_symbols()
        print(f"[INIT] Loaded {len(self.symbols)} symbols for scanning")
        
        # Initialize components
        self.screener = HistoricalParabolicScreener()
        self.ml_risk = InstitutionalRiskManager()
        self.v5_strategy = get_strategy('v5_relaxed_scanner')
        
        # Statistics tracking
        self.stats = {
            'days_processed': 0,
            'symbols_scanned': 0,
            'setups_found': 0,
            'v5_trades_taken': 0,
            'v5_pnl': 0.0,
            'ml_trades_taken': 0,
            'ml_trades_blocked': 0,
            'ml_pnl': 0.0,
            'current_symbol': '',
            'current_date': ''
        }
        
        # Results storage
        self.trades: List[TradeRecord] = []
        
        # Create output directory
        self.output_dir = Path('reports/complete_fresh_backtest')
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Start time
        self.start_time = time.time()
        
    def _load_all_symbols(self) -> List[str]:
        """Load the full 3,527 symbol universe."""
        try:
            # Try to load from extended universe
            from src.backtest.extended_universe import EXTENDED_UNIVERSE
            return EXTENDED_UNIVERSE
        except:
            pass
        
        # Fallback: scan all CSV files in data directory
        symbol_files = list(Path('data/cache').glob('*_trades_*.parquet'))
        symbols = set()
        for f in symbol_files:
            symbol = f.name.split('_trades_')[0]
            symbols.add(symbol)
        
        if len(symbols) > 100:
            return list(symbols)
        
        # Last resort: use our backtest results
        df = pd.read_csv('reports/full_3527_backtest_results.csv')
        return df['symbol'].unique().tolist()
    
    def run_complete_backtest(self):
        """Run the complete fresh backtest."""
        print("\n" + "="*80)
        print("COMPLETE FRESH BACKTEST - FULL 6-YEAR SCAN")
        print("="*80)
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Symbols: {len(self.symbols)}")
        print(f"Estimated Setups: ~900-1000")
        print(f"Estimated Runtime: 6-8 hours")
        print("="*80 + "\n")
        
        # Process each symbol
        for idx, symbol in enumerate(self.symbols):
            self.stats['symbols_scanned'] += 1
            self.stats['current_symbol'] = symbol
            
            # Progress update
            if idx % 10 == 0:
                self._print_progress(idx)
            
            # Process this symbol across all dates
            self._process_symbol(symbol)
            
            # Periodic checkpoint save
            if idx % 50 == 0 and idx > 0:
                self._save_checkpoint()
        
        # Final save
        self._save_results()
        self._generate_report()
        
    def _process_symbol(self, symbol: str):
        """Process a single symbol across all dates."""
        current_date = self.start_date
        
        while current_date <= self.end_date:
            # Skip weekends
            if current_date.weekday() < 5:
                self.stats['days_processed'] += 1
                self.stats['current_date'] = current_date.strftime('%Y-%m-%d')
                
                # Check if this symbol/date is a parabolic setup
                setup = self._check_setup(symbol, current_date)
                
                if setup:
                    self.stats['setups_found'] += 1
                    self._process_setup(setup)
            
            current_date += timedelta(days=1)
    
    def _check_setup(self, symbol: str, date: datetime) -> Optional[Dict]:
        """Check if symbol/date is a parabolic setup."""
        try:
            # Fetch tick data
            tick_df = tick_fetcher.fetch_combined_tick_data(symbol, date, use_quotes=False)
            if tick_df.is_empty():
                return None
            
            # Aggregate to bars
            bar_df = tick_fetcher.aggregate_trades_to_bars(tick_df, interval_seconds=60)
            if bar_df.is_empty() or len(bar_df) < 30:
                return None
            
            bars = bar_df.to_pandas()
            bars['timestamp'] = pd.to_datetime(bars['timestamp'])
            bars = bars.sort_values('timestamp')
            
            # Calculate metrics
            day_open = bars.iloc[0]['open']
            day_high = bars['high'].max()
            day_low = bars['low'].min()
            day_gain = (day_high - day_open) / day_open * 100
            
            # Relaxed scanner criteria (30% gain threshold)
            if day_gain < 30:
                return None
            
            # Calculate VWAP
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
                'day_low': day_low,
                'day_gain': day_gain
            }
            
        except Exception as e:
            return None
    
    def _process_setup(self, setup: Dict):
        """Process a found setup with both strategies."""
        symbol = setup['symbol']
        date = setup['date']
        date_str = date.strftime('%Y-%m-%d')
        bars = setup['bars']
        
        # Calculate features for ML
        features = self._extract_features(bars)
        
        # === V5 RELAXED STRATEGY ===
        # Run V5 backtest (always takes the trade)
        try:
            v5_result = self.v5_strategy.run_tick_backtest(symbol, date, verbose=False)
            
            if v5_result.total_trades > 0:
                self.stats['v5_trades_taken'] += 1
                self.stats['v5_pnl'] += v5_result.total_pnl
                
                trade_v5 = TradeRecord(
                    symbol=symbol,
                    date=date_str,
                    day_gain_pct=setup['day_gain'],
                    entry_price=v5_result.trades[0].entry_price if hasattr(v5_result, 'trades') else setup['day_open'],
                    exit_price=v5_result.trades[0].exit_price if hasattr(v5_result, 'trades') else setup['day_high'],
                    shares=0,
                    pnl=v5_result.total_pnl,
                    win=1 if v5_result.total_pnl > 0 else 0,
                    strategy='v5_relaxed',
                    ml_blocked=False,
                    **features
                )
                self.trades.append(trade_v5)
        except:
            pass
        
        # === V5 INSTITUTIONAL ML STRATEGY ===
        # Run ML risk assessment
        try:
            raw_data = {
                'symbol': symbol,
                'date': date_str,
                'bars': bars.to_dict('records')
            }
            
            assessment = self.ml_risk.assess_trade(raw_data)
            
            if assessment['recommendation'] == 'AVOID':
                # Blocked by ML
                self.stats['ml_trades_blocked'] += 1
                
                trade_ml = TradeRecord(
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
                )
                self.trades.append(trade_ml)
                
            else:
                # Approved - take the trade with Kelly sizing
                kelly = assessment['kelly_fraction']
                
                # Simulate trade (simplified)
                # In reality would run full backtest, here we estimate
                entry_bar = bars[(bars['timestamp'].dt.hour >= 10) & (bars['timestamp'].dt.hour < 11)]
                if entry_bar.empty:
                    entry_price = bars.iloc[30]['close']  # ~10:00 AM
                else:
                    entry_price = entry_bar.iloc[0]['close']
                
                vwap = bars['vwap'].iloc[-1]
                exit_price = vwap  # Target VWAP
                
                # Position size based on Kelly
                position_value = 25000 * kelly
                shares = int(position_value / entry_price)
                
                # Calculate P&L
                pnl = (entry_price - exit_price) * shares
                
                self.stats['ml_trades_taken'] += 1
                self.stats['ml_pnl'] += pnl
                
                # Update ML with outcome
                self.ml_risk.update_online({
                    'predicted_win_prob': assessment['win_probability'],
                    'actual_outcome': 1 if pnl > 0 else 0,
                    'actual_pnl': pnl
                })
                
                trade_ml = TradeRecord(
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
                )
                self.trades.append(trade_ml)
                
        except Exception as e:
            print(f"  ML error on {symbol} {date_str}: {e}")
    
    def _extract_features(self, bars: pd.DataFrame) -> Dict:
        """Extract key features from bars."""
        try:
            # Time to peak
            peak_idx = bars['high'].idxmax()
            peak_time = bars.loc[peak_idx, 'timestamp']
            market_open = bars.iloc[0]['timestamp']
            minutes_to_peak = (peak_time - market_open).total_seconds() / 60
            
            # VWAP deviation at peak
            peak_price = bars.loc[peak_idx, 'high']
            vwap_at_peak = bars.loc[peak_idx, 'vwap']
            vwap_deviation = ((peak_price - vwap_at_peak) / vwap_at_peak) * 100
            
            # Volume concentration
            first_hour_mask = bars['timestamp'].dt.hour < 11
            first_hour_vol = bars[first_hour_mask]['volume'].sum() if first_hour_mask.any() else bars.head(30)['volume'].sum()
            total_vol = bars['volume'].sum()
            volume_conc = first_hour_vol / total_vol if total_vol > 0 else 0
            
            return {
                'minutes_to_peak': minutes_to_peak,
                'vwap_deviation': vwap_deviation,
                'volume_concentration': volume_conc
            }
        except:
            return {
                'minutes_to_peak': 0,
                'vwap_deviation': 0,
                'volume_concentration': 0
            }
    
    def _print_progress(self, current_idx: int):
        """Print progress update."""
        elapsed = time.time() - self.start_time
        pct = (current_idx / len(self.symbols)) * 100
        
        if current_idx > 0:
            time_per_symbol = elapsed / current_idx
            remaining = time_per_symbol * (len(self.symbols) - current_idx)
            eta = timedelta(seconds=int(remaining))
        else:
            eta = "Calculating..."
        
        print(f"\n{'='*80}")
        print(f"PROGRESS: {current_idx}/{len(self.symbols)} symbols ({pct:.1f}%)")
        print(f"ETA: {eta}")
        print(f"Current: {self.stats['current_symbol']} on {self.stats['current_date']}")
        print("-"*80)
        print(f"Days Processed: {self.stats['days_processed']}")
        print(f"Setups Found: {self.stats['setups_found']}")
        print(f"V5: {self.stats['v5_trades_taken']} trades, ${self.stats['v5_pnl']:,.0f} P&L")
        print(f"ML: {self.stats['ml_trades_taken']} trades, {self.stats['ml_trades_blocked']} blocked, ${self.stats['ml_pnl']:,.0f} P&L")
        print(f"{'='*80}\n")
    
    def _save_checkpoint(self):
        """Save checkpoint."""
        if len(self.trades) > 0:
            df = pd.DataFrame([t.to_dict() for t in self.trades])
            df.to_csv(self.output_dir / 'checkpoint.csv', index=False)
            
            with open(self.output_dir / 'stats.json', 'w') as f:
                json.dump(self.stats, f, indent=2)
    
    def _save_results(self):
        """Save final results."""
        print("\n[SAVING] Saving results...")
        
        df = pd.DataFrame([t.to_dict() for t in self.trades])
        df.to_csv(self.output_dir / 'complete_trades.csv', index=False)
        
        with open(self.output_dir / 'final_stats.json', 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"  Saved {len(df)} trade records")
    
    def _generate_report(self):
        """Generate final report."""
        print("\n[REPORT] Generating comparison report...")
        
        df = pd.DataFrame([t.to_dict() for t in self.trades])
        
        # Calculate statistics
        v5_df = df[df['strategy'] == 'v5_relaxed']
        inst_df = df[df['strategy'] == 'v5_institutional']
        
        v5_trades = v5_df[v5_df['pnl'] != 0]
        inst_taken = inst_df[inst_df['ml_blocked'] == False]
        inst_blocked = inst_df[inst_df['ml_blocked'] == True]
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'period': f"{self.start_date.date()} to {self.end_date.date()}",
            'symbols_scanned': self.stats['symbols_scanned'],
            'days_processed': self.stats['days_processed'],
            'total_setups_found': self.stats['setups_found'],
            
            'v5_relaxed': {
                'trades_taken': len(v5_trades),
                'win_rate': float(v5_trades['win'].mean() * 100) if len(v5_trades) > 0 else 0,
                'total_pnl': float(v5_trades['pnl'].sum()),
                'avg_trade': float(v5_trades['pnl'].mean()) if len(v5_trades) > 0 else 0,
                'avg_win': float(v5_trades[v5_trades['pnl'] > 0]['pnl'].mean()) if len(v5_trades[v5_trades['pnl'] > 0]) > 0 else 0,
                'avg_loss': float(v5_trades[v5_trades['pnl'] < 0]['pnl'].mean()) if len(v5_trades[v5_trades['pnl'] < 0]) > 0 else 0,
            },
            
            'v5_institutional': {
                'trades_taken': len(inst_taken),
                'trades_blocked': len(inst_blocked),
                'block_rate': float(len(inst_blocked) / len(inst_df) * 100) if len(inst_df) > 0 else 0,
                'win_rate': float(inst_taken['win'].mean() * 100) if len(inst_taken) > 0 else 0,
                'total_pnl': float(inst_taken['pnl'].sum()),
                'avg_trade': float(inst_taken['pnl'].mean()) if len(inst_taken) > 0 else 0,
                'avg_win': float(inst_taken[inst_taken['pnl'] > 0]['pnl'].mean()) if len(inst_taken[inst_taken['pnl'] > 0]) > 0 else 0,
                'avg_loss': float(inst_taken[inst_taken['pnl'] < 0]['pnl'].mean()) if len(inst_taken[inst_taken['pnl'] < 0]) > 0 else 0,
                'avg_risk_score': float(inst_taken['risk_score'].mean()) if len(inst_taken) > 0 else 0,
                'avg_win_probability': float(inst_taken['win_probability'].mean()) if len(inst_taken) > 0 else 0,
            }
        }
        
        # Comparison
        pnl_diff = report['v5_institutional']['total_pnl'] - report['v5_relaxed']['total_pnl']
        wr_diff = report['v5_institutional']['win_rate'] - report['v5_relaxed']['win_rate']
        
        report['comparison'] = {
            'pnl_difference': pnl_diff,
            'win_rate_difference': wr_diff,
            'recommendation': 'V5 Institutional' if pnl_diff > 0 else 'V5 Relaxed'
        }
        
        with open(self.output_dir / 'report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("BACKTEST COMPLETE - FINAL RESULTS")
        print("="*80)
        print(f"\nPeriod: {report['period']}")
        print(f"Symbols Scanned: {report['symbols_scanned']}")
        print(f"Setups Found: {report['total_setups_found']}")
        
        print("\n--- V5 RELAXED SCANNER ---")
        print(f"Trades Taken: {report['v5_relaxed']['trades_taken']}")
        print(f"Win Rate: {report['v5_relaxed']['win_rate']:.1f}%")
        print(f"Total P&L: ${report['v5_relaxed']['total_pnl']:,.2f}")
        print(f"Average Trade: ${report['v5_relaxed']['avg_trade']:,.2f}")
        
        print("\n--- V5 INSTITUTIONAL ML ---")
        print(f"Trades Taken: {report['v5_institutional']['trades_taken']}")
        print(f"Trades Blocked: {report['v5_institutional']['trades_blocked']} ({report['v5_institutional']['block_rate']:.1f}%)")
        print(f"Win Rate: {report['v5_institutional']['win_rate']:.1f}%")
        print(f"Total P&L: ${report['v5_institutional']['total_pnl']:,.2f}")
        print(f"Average Trade: ${report['v5_institutional']['avg_trade']:,.2f}")
        
        print("\n--- COMPARISON ---")
        print(f"P&L Difference: ${pnl_diff:+,.2f}")
        print(f"Win Rate Difference: {wr_diff:+.1f}%")
        print(f"Winner: {report['comparison']['recommendation']}")
        
        print("\n" + "="*80)
        print(f"Results saved to: {self.output_dir}")
        print("="*80)


def main():
    print("="*80)
    print("COMPLETE FRESH BACKTEST - FULL SCAN WITH ML RISK ENGINE")
    print("="*80)
    print("\nThis will:")
    print("  1. Scan ALL 3,527 symbols")
    print("  2. Process ALL trading days from 2019-2024")
    print("  3. Find parabolic setups (30%+ gain)")
    print("  4. Run ML risk assessment on each setup")
    print("  5. Compare V5 Relaxed vs V5 Institutional")
    print("\nEstimated time: 6-8 hours")
    print("="*80)
    
    response = input("\nStart complete backtest? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted.")
        return
    
    # Run
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    engine = FreshBacktestEngine(start_date, end_date)
    engine.run_complete_backtest()


if __name__ == "__main__":
    main()
