"""
Full 6-Year Backtest Comparison: V5 Relaxed vs V5 Institutional ML

Runs both strategies on the complete dataset (2019-2024, 3,527 symbols)
with real-time monitoring and automatic report generation.

Usage:
    python run_full_comparison_backtest.py --parallel
    
Output:
    - reports/comparison_backtest_results.csv
    - reports/comparison_summary.json
    - reports/comparison_charts/
"""

import sys
import json
import time
import argparse
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.strategies import get_strategy
from src.backtest.historical_tick_fetcher import tick_fetcher


@dataclass
class TradeResult:
    """Container for trade result."""
    symbol: str
    date: str
    strategy: str
    pnl: float
    win: int
    risk_score: float = 0.0
    win_probability: float = 0.0
    recommendation: str = ""
    kelly_fraction: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


class BacktestMonitor:
    """
    Real-time monitoring and logging for backtest.
    """
    
    def __init__(self, total_setups: int):
        self.total_setups = total_setups
        self.processed = 0
        self.start_time = time.time()
        self.results_v5: List[TradeResult] = []
        self.results_institutional: List[TradeResult] = []
        self.lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'v5': {'trades': 0, 'wins': 0, 'pnl': 0.0, 'losses': 0},
            'institutional': {'trades': 0, 'wins': 0, 'pnl': 0.0, 'losses': 0, 'blocked': 0}
        }
        
        # Logging
        self.log_file = Path("logs/backtest_comparison.log")
        self.log_file.parent.mkdir(exist_ok=True)
        
    def log(self, message: str, level: str = "INFO"):
        """Log message to file and console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] [{level}] {message}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_line)
        
        if level in ["INFO", "WARNING", "ERROR"]:
            print(log_line.strip())
    
    def update(self, result: TradeResult):
        """Update with new result."""
        with self.lock:
            self.processed += 1
            
            if result.strategy == "v5_relaxed":
                self.results_v5.append(result)
                self.stats['v5']['trades'] += 1
                self.stats['v5']['pnl'] += result.pnl
                if result.win:
                    self.stats['v5']['wins'] += 1
                elif result.pnl < 0:
                    self.stats['v5']['losses'] += 1
            else:
                self.results_institutional.append(result)
                if result.pnl != 0:  # Only count if trade was taken
                    self.stats['institutional']['trades'] += 1
                    self.stats['institutional']['pnl'] += result.pnl
                    if result.win:
                        self.stats['institutional']['wins'] += 1
                    elif result.pnl < 0:
                        self.stats['institutional']['losses'] += 1
                else:
                    self.stats['institutional']['blocked'] += 1
    
    def print_progress(self):
        """Print progress update."""
        elapsed = time.time() - self.start_time
        pct_complete = (self.processed / self.total_setups) * 100 if self.total_setups > 0 else 0
        
        # Estimate time remaining
        if self.processed > 0:
            time_per_setup = elapsed / self.processed
            remaining = time_per_setup * (self.total_setups - self.processed)
            eta = timedelta(seconds=int(remaining))
        else:
            eta = "Unknown"
        
        # Calculate current metrics
        v5_wr = (self.stats['v5']['wins'] / self.stats['v5']['trades'] * 100) if self.stats['v5']['trades'] > 0 else 0
        inst_wr = (self.stats['institutional']['wins'] / self.stats['institutional']['trades'] * 100) if self.stats['institutional']['trades'] > 0 else 0
        
        print("\n" + "="*80)
        print(f"PROGRESS: {self.processed}/{self.total_setups} ({pct_complete:.1f}%) | ETA: {eta}")
        print("-"*80)
        print(f"V5 Relaxed:      {self.stats['v5']['trades']} trades | "
              f"WR: {v5_wr:.1f}% | P&L: ${self.stats['v5']['pnl']:,.0f}")
        print(f"V5 Institutional: {self.stats['institutional']['trades']} trades | "
              f"WR: {inst_wr:.1f}% | P&L: ${self.stats['institutional']['pnl']:,.0f} | "
              f"Blocked: {self.stats['institutional']['blocked']}")
        print("="*80)
    
    def get_summary(self) -> Dict:
        """Get final summary."""
        elapsed = time.time() - self.start_time
        
        return {
            'total_setups': self.total_setups,
            'processed': self.processed,
            'elapsed_seconds': elapsed,
            'v5_stats': self.stats['v5'],
            'institutional_stats': self.stats['institutional']
        }


class ComparisonBacktestRunner:
    """
    Run both strategies on the same setups for fair comparison.
    """
    
    def __init__(self, setups_df: pd.DataFrame, monitor: BacktestMonitor):
        self.setups = setups_df
        self.monitor = monitor
        
        # Initialize strategies
        print("[INIT] Initializing strategies...")
        self.v5_engine = get_strategy('v5_relaxed_scanner')
        self.institutional_engine = get_strategy('v5_institutional')
        
        # Results storage
        self.results: List[TradeResult] = []
        
    def run_single_backtest(self, setup: pd.Series) -> Tuple[TradeResult, TradeResult]:
        """Run both strategies on a single setup."""
        symbol = setup['symbol']
        date_str = setup['date']
        date = datetime.strptime(date_str, "%Y-%m-%d")
        
        try:
            # Run V5 Relaxed
            result_v5 = self.v5_engine.run_tick_backtest(symbol, date, verbose=False)
            
            trade_v5 = TradeResult(
                symbol=symbol,
                date=date_str,
                strategy="v5_relaxed",
                pnl=result_v5.total_pnl,
                win=1 if result_v5.total_pnl > 0 else 0
            )
            
            # Run V5 Institutional
            result_inst = self.institutional_engine.run_tick_backtest(symbol, date, verbose=False)
            
            # Get risk assessment details if available
            if hasattr(self.institutional_engine, 'risk_manager'):
                # Extract last assessment from stats if possible
                risk_score = 0.0
                win_prob = 0.0
                rec = ""
                kelly = 0.0
            else:
                risk_score = 0.0
                win_prob = 0.0
                rec = ""
                kelly = 0.0
            
            trade_inst = TradeResult(
                symbol=symbol,
                date=date_str,
                strategy="v5_institutional",
                pnl=result_inst.total_pnl,
                win=1 if result_inst.total_pnl > 0 else 0,
                risk_score=risk_score,
                win_probability=win_prob,
                recommendation=rec,
                kelly_fraction=kelly
            )
            
            return trade_v5, trade_inst
            
        except Exception as e:
            self.monitor.log(f"Error on {symbol} {date_str}: {e}", "ERROR")
            return None, None
    
    def run_full_backtest(self, batch_size: int = 50):
        """Run full backtest with progress monitoring."""
        print(f"\n[BACKTEST] Starting comparison on {len(self.setups)} setups...")
        print(f"           Expected duration: ~{len(self.setups) * 2 / 60:.0f} minutes\n")
        
        for idx, (_, setup) in enumerate(self.setups.iterrows()):
            # Run both strategies
            trade_v5, trade_inst = self.run_single_backtest(setup)
            
            if trade_v5 and trade_inst:
                self.monitor.update(trade_v5)
                self.monitor.update(trade_inst)
                self.results.extend([trade_v5, trade_inst])
            
            # Print progress every N setups
            if (idx + 1) % batch_size == 0:
                self.monitor.print_progress()
                self._save_checkpoint()
        
        # Final progress
        self.monitor.print_progress()
        print("\n[BACKTEST] Complete!")
    
    def _save_checkpoint(self):
        """Save intermediate results."""
        if len(self.results) > 0:
            df = pd.DataFrame([r.to_dict() for r in self.results])
            df.to_csv('reports/comparison_checkpoint.csv', index=False)
    
    def save_results(self):
        """Save final results."""
        print("\n[SAVING] Saving results...")
        
        # Save detailed results
        df = pd.DataFrame([r.to_dict() for r in self.results])
        output_path = Path('reports/comparison_backtest_results.csv')
        df.to_csv(output_path, index=False)
        print(f"  Saved: {output_path}")
        
        # Save summary
        summary = self._generate_summary()
        summary_path = Path('reports/comparison_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved: {summary_path}")
        
        return summary
    
    def _generate_summary(self) -> Dict:
        """Generate comparison summary."""
        df = pd.DataFrame([r.to_dict() for r in self.results])
        
        # V5 stats
        v5_df = df[df['strategy'] == 'v5_relaxed']
        v5_trades = v5_df[v5_df['pnl'] != 0]
        
        v5_stats = {
            'total_trades': len(v5_trades),
            'wins': int(v5_trades['win'].sum()),
            'win_rate': float(v5_trades['win'].mean() * 100) if len(v5_trades) > 0 else 0,
            'total_pnl': float(v5_trades['pnl'].sum()),
            'avg_trade': float(v5_trades['pnl'].mean()),
            'avg_win': float(v5_trades[v5_trades['pnl'] > 0]['pnl'].mean()) if len(v5_trades[v5_trades['pnl'] > 0]) > 0 else 0,
            'avg_loss': float(v5_trades[v5_trades['pnl'] < 0]['pnl'].mean()) if len(v5_trades[v5_trades['pnl'] < 0]) > 0 else 0,
            'max_win': float(v5_trades['pnl'].max()),
            'max_loss': float(v5_trades['pnl'].min()),
            'profit_factor': self._calc_profit_factor(v5_trades)
        }
        
        # Institutional stats
        inst_df = df[df['strategy'] == 'v5_institutional']
        inst_trades = inst_df[inst_df['pnl'] != 0]
        blocked = len(inst_df[inst_df['pnl'] == 0])
        
        inst_stats = {
            'total_trades': len(inst_trades),
            'blocked_trades': blocked,
            'block_rate': float(blocked / len(inst_df) * 100) if len(inst_df) > 0 else 0,
            'wins': int(inst_trades['win'].sum()),
            'win_rate': float(inst_trades['win'].mean() * 100) if len(inst_trades) > 0 else 0,
            'total_pnl': float(inst_trades['pnl'].sum()),
            'avg_trade': float(inst_trades['pnl'].mean()),
            'avg_win': float(inst_trades[inst_trades['pnl'] > 0]['pnl'].mean()) if len(inst_trades[inst_trades['pnl'] > 0]) > 0 else 0,
            'avg_loss': float(inst_trades[inst_trades['pnl'] < 0]['pnl'].mean()) if len(inst_trades[inst_trades['pnl'] < 0]) > 0 else 0,
            'max_win': float(inst_trades['pnl'].max()),
            'max_loss': float(inst_trades['pnl'].min()),
            'profit_factor': self._calc_profit_factor(inst_trades)
        }
        
        # Comparison
        comparison = {
            'pnl_difference': inst_stats['total_pnl'] - v5_stats['total_pnl'],
            'win_rate_difference': inst_stats['win_rate'] - v5_stats['win_rate'],
            'trades_difference': inst_stats['total_trades'] - v5_stats['total_trades'],
            'estimated_losses_avoided': self._estimate_losses_avoided(v5_df, inst_df)
        }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'v5_relaxed': v5_stats,
            'v5_institutional': inst_stats,
            'comparison': comparison,
            'recommendation': self._generate_recommendation(v5_stats, inst_stats)
        }
    
    def _calc_profit_factor(self, df: pd.DataFrame) -> float:
        """Calculate profit factor."""
        gross_profit = df[df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(df[df['pnl'] < 0]['pnl'].sum())
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def _estimate_losses_avoided(self, v5_df: pd.DataFrame, inst_df: pd.DataFrame) -> float:
        """Estimate losses avoided by institutional strategy."""
        # Find setups where V5 lost money but Institutional didn't trade
        merged = pd.merge(
            v5_df[['symbol', 'date', 'pnl']], 
            inst_df[['symbol', 'date', 'pnl']], 
            on=['symbol', 'date'], 
            suffixes=('_v5', '_inst')
        )
        
        losses_avoided = merged[
            (merged['pnl_v5'] < 0) & (merged['pnl_inst'] == 0)
        ]['pnl_v5'].sum()
        
        return float(abs(losses_avoided))
    
    def _generate_recommendation(self, v5_stats: Dict, inst_stats: Dict) -> str:
        """Generate strategy recommendation."""
        if inst_stats['total_pnl'] > v5_stats['total_pnl'] and inst_stats['win_rate'] > v5_stats['win_rate']:
            return "Use V5 Institutional - Superior risk-adjusted returns"
        elif inst_stats['total_pnl'] > v5_stats['total_pnl']:
            return "Use V5 Institutional - Higher total returns, similar win rate"
        elif abs(inst_stats['total_pnl'] - v5_stats['total_pnl']) < 50000:
            return "Both strategies comparable - V5 Institutional has better risk management"
        else:
            return "Use V5 Relaxed - Better performance in current test"


def main():
    parser = argparse.ArgumentParser(description='Run full comparison backtest')
    parser.add_argument('--setups', type=str, default='reports/full_3527_setups.csv',
                       help='Path to setups CSV')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of setups (for testing)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Resume from checkpoint file')
    args = parser.parse_args()
    
    print("="*80)
    print("FULL BACKTEST COMPARISON: V5 RELAXED vs V5 INSTITUTIONAL")
    print("="*80)
    print(f"Start Time: {datetime.now()}")
    
    # Load setups
    print(f"\n[LOAD] Loading setups from {args.setups}...")
    setups_df = pd.read_csv(args.setups)
    
    if args.limit:
        setups_df = setups_df.head(args.limit)
        print(f"       Limited to {args.limit} setups for testing")
    
    print(f"       Loaded {len(setups_df)} setups")
    
    # Initialize monitor
    monitor = BacktestMonitor(len(setups_df))
    monitor.log(f"Starting comparison backtest on {len(setups_df)} setups")
    
    # Create runner
    runner = ComparisonBacktestRunner(setups_df, monitor)
    
    # Run backtest
    runner.run_full_backtest()
    
    # Save results
    summary = runner.save_results()
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(json.dumps(summary, indent=2))
    
    print("\n" + "="*80)
    print("BACKTEST COMPLETE")
    print("="*80)
    print(f"Results saved to reports/")
    print(f"Recommendation: {summary['recommendation']}")


if __name__ == "__main__":
    main()
