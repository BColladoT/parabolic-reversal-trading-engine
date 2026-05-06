"""
Quick Test - 100 Symbols with ML Risk Engine

This runs a quick test on just the first 100 symbols to verify everything works
before committing to the full 8-12 hour backtest.
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
from src.backtest.extended_universe import EXTENDED_MICRO_CAP_SYMBOLS
from src.risk.ml_simple import InstitutionalRiskManager
from src.strategies import get_strategy


# Use ONLY first 100 symbols for quick test
TEST_SYMBOLS = EXTENDED_MICRO_CAP_SYMBOLS[:100]
print(f"Quick test on {len(TEST_SYMBOLS)} symbols")


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


class QuickBacktestEngine:
    """Quick test engine using only 100 symbols."""
    
    def __init__(self, start_date: datetime, end_date: datetime):
        self.start_date = start_date
        self.end_date = end_date
        self.symbols = TEST_SYMBOLS
        
        print(f"\n[INIT] Quick test on {len(self.symbols)} symbols")
        
        self.ml_risk = InstitutionalRiskManager()
        self.v5_strategy = get_strategy('v5_relaxed_scanner')
        
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
        
        self.trades: List[TradeRecord] = []
        self.output_dir = Path('reports/quick_100_backtest')
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.start_time = time.time()
        
    def run_complete_backtest(self):
        """Run the quick backtest."""
        print("\n" + "="*80)
        print("QUICK TEST - 100 SYMBOLS WITH ML RISK ENGINE")
        print("="*80)
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Symbols: {len(self.symbols)}")
        print(f"Estimated Runtime: 20-30 minutes")
        print("="*80 + "\n")
        
        for idx, symbol in enumerate(self.symbols):
            self.stats['symbols_scanned'] += 1
            self.stats['current_symbol'] = symbol
            
            if idx % 10 == 0:
                self._print_progress(idx)
            
            self._process_symbol(symbol)
            
            if idx % 20 == 0 and idx > 0:
                self._save_checkpoint()
        
        self._save_results()
        self._generate_report()
        
    def _process_symbol(self, symbol: str):
        """Process a single symbol across all dates."""
        current_date = self.start_date
        
        while current_date <= self.end_date:
            if current_date.weekday() < 5:
                self.stats['days_processed'] += 1
                self.stats['current_date'] = current_date.strftime('%Y-%m-%d')
                
                setup = self._check_setup(symbol, current_date)
                
                if setup:
                    self.stats['setups_found'] += 1
                    self._process_setup(setup)
            
            current_date += timedelta(days=1)
    
    def _check_setup(self, symbol: str, date: datetime) -> Optional[Dict]:
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
    
    def _process_setup(self, setup: Dict):
        """Process a found setup with both strategies."""
        symbol = setup['symbol']
        date = setup['date']
        date_str = date.strftime('%Y-%m-%d')
        bars = setup['bars']
        
        features = self._extract_features(bars)
        
        # V5 RELAXED
        try:
            v5_result = self.v5_strategy.run_tick_backtest(symbol, date, verbose=False)
            
            if v5_result.total_trades > 0:
                self.stats['v5_trades_taken'] += 1
                self.stats['v5_pnl'] += v5_result.total_pnl
                
                trade_v5 = TradeRecord(
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
                )
                self.trades.append(trade_v5)
        except:
            pass
        
        # V5 INSTITUTIONAL ML
        try:
            raw_data = {
                'symbol': symbol,
                'date': date_str,
                'bars': bars.to_dict('records')
            }
            
            assessment = self.ml_risk.assess_trade(raw_data)
            
            if assessment['recommendation'] == 'AVOID':
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
                kelly = assessment['kelly_fraction']
                
                entry_bar = bars[(bars['timestamp'].dt.hour >= 10) & (bars['timestamp'].dt.hour < 11)]
                entry_price = entry_bar.iloc[0]['close'] if not entry_bar.empty else bars.iloc[30]['close']
                vwap = bars['vwap'].iloc[-1]
                exit_price = vwap
                
                position_value = 25000 * kelly
                shares = int(position_value / entry_price)
                pnl = (entry_price - exit_price) * shares
                
                self.stats['ml_trades_taken'] += 1
                self.stats['ml_pnl'] += pnl
                
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
            pass
    
    def _extract_features(self, bars: pd.DataFrame) -> Dict:
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
            
            return {
                'minutes_to_peak': minutes_to_peak,
                'vwap_deviation': vwap_deviation,
                'volume_concentration': volume_conc
            }
        except:
            return {'minutes_to_peak': 0, 'vwap_deviation': 0, 'volume_concentration': 0}
    
    def _print_progress(self, current_idx: int):
        """Print progress update."""
        elapsed = time.time() - self.start_time
        pct = (current_idx / len(self.symbols)) * 100
        
        if current_idx > 0:
            time_per_symbol = elapsed / current_idx
            remaining = time_per_symbol * (len(self.symbols) - current_idx)
            eta_mins = remaining / 60
        else:
            eta_mins = 0
        
        print(f"\n{'='*80}")
        print(f"PROGRESS: {current_idx}/{len(self.symbols)} symbols ({pct:.1f}%)")
        print(f"ETA: {eta_mins:.0f} minutes remaining")
        print("-"*80)
        print(f"Setups Found: {self.stats['setups_found']}")
        print(f"V5: {self.stats['v5_trades_taken']} trades, ${self.stats['v5_pnl']:,.0f}")
        print(f"ML: {self.stats['ml_trades_taken']} taken, {self.stats['ml_trades_blocked']} blocked, ${self.stats['ml_pnl']:,.0f}")
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
    
    def _generate_report(self):
        """Generate final report."""
        print("\n[REPORT] Generating comparison report...")
        df = pd.DataFrame([t.to_dict() for t in self.trades])
        
        v5_df = df[df['strategy'] == 'v5_relaxed']
        inst_df = df[df['strategy'] == 'v5_institutional']
        
        v5_trades = v5_df[v5_df['pnl'] != 0]
        inst_taken = inst_df[inst_df['ml_blocked'] == False]
        inst_blocked = inst_df[inst_df['ml_blocked'] == True]
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'period': f"{self.start_date.date()} to {self.end_date.date()}",
            'symbols_scanned': len(self.symbols),
            'days_processed': self.stats['days_processed'],
            'total_setups_found': self.stats['setups_found'],
            
            'v5_relaxed': {
                'trades_taken': len(v5_trades),
                'win_rate': float(v5_trades['win'].mean() * 100) if len(v5_trades) > 0 else 0,
                'total_pnl': float(v5_trades['pnl'].sum()),
                'avg_trade': float(v5_trades['pnl'].mean()) if len(v5_trades) > 0 else 0,
            },
            
            'v5_institutional': {
                'trades_taken': len(inst_taken),
                'trades_blocked': len(inst_blocked),
                'block_rate': float(len(inst_blocked) / len(inst_df) * 100) if len(inst_df) > 0 else 0,
                'win_rate': float(inst_taken['win'].mean() * 100) if len(inst_taken) > 0 else 0,
                'total_pnl': float(inst_taken['pnl'].sum()),
                'avg_trade': float(inst_taken['pnl'].mean()) if len(inst_taken) > 0 else 0,
            }
        }
        
        pnl_diff = report['v5_institutional']['total_pnl'] - report['v5_relaxed']['total_pnl']
        wr_diff = report['v5_institutional']['win_rate'] - report['v5_relaxed']['win_rate']
        
        report['comparison'] = {
            'pnl_difference': pnl_diff,
            'win_rate_difference': wr_diff,
            'recommendation': 'V5 Institutional' if pnl_diff > 0 else 'V5 Relaxed'
        }
        
        with open(self.output_dir / 'report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\n" + "="*80)
        print("QUICK TEST COMPLETE - RESULTS")
        print("="*80)
        print(f"Symbols: {report['symbols_scanned']}, Setups: {report['total_setups_found']}")
        print(f"V5: {report['v5_relaxed']['trades_taken']} trades, ${report['v5_relaxed']['total_pnl']:,.0f}")
        print(f"ML: {report['v5_institutional']['trades_taken']} taken, {report['v5_institutional']['trades_blocked']} blocked, ${report['v5_institutional']['total_pnl']:,.0f}")
        print(f"Winner: {report['comparison']['recommendation']} (P&L diff: ${pnl_diff:+,.0f})")
        print("="*80)


def main():
    print("="*80)
    print("QUICK 100 SYMBOL BACKTEST WITH ML RISK ENGINE")
    print("="*80)
    print(f"\nSymbols: {len(TEST_SYMBOLS)}")
    print(f"Period: 2019-01-01 to 2024-12-31")
    print(f"Estimated Runtime: 20-30 minutes")
    
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    engine = QuickBacktestEngine(start_date, end_date)
    engine.run_complete_backtest()


if __name__ == "__main__":
    main()
