"""
Live Trading Simulation with ML Risk Management

Comprehensive backtest that:
1. Re-scans all 3,527 symbols for parabolic setups (2019-2024)
2. Simulates real-time trading with both strategies
3. ML risk engine makes live decisions on each trade
4. Tracks portfolio, positions, and P&L in real-time

Usage:
    python run_live_trading_simulation.py --start-date 2019-01-01 --end-date 2024-12-31
    
Output:
    - reports/live_simulation_portfolio.csv
    - reports/live_simulation_trades.csv
    - reports/live_simulation_daily_pnl.csv
    - reports/live_simulation_report.html
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import pandas as pd
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.strategies import get_strategy
from src.backtest.historical_screener import HistoricalParabolicScreener
from src.backtest.historical_tick_fetcher import tick_fetcher
from src.risk.ml_simple import InstitutionalRiskManager


@dataclass
class Position:
    """Live position tracking."""
    symbol: str
    entry_date: datetime
    entry_price: float
    shares: int
    position_value: float
    stop_loss: float
    profit_target: float
    strategy: str
    risk_score: float = 0.0
    win_probability: float = 0.0
    
    def current_pnl(self, current_price: float) -> float:
        """Calculate current P&L."""
        return (self.entry_price - current_price) * self.shares


@dataclass
class Portfolio:
    """Portfolio state tracking."""
    cash: float = 100000.0
    positions: Dict[str, Position] = field(default_factory=dict)
    daily_pnl: List[Dict] = field(default_factory=list)
    closed_trades: List[Dict] = field(default_factory=list)
    
    def total_equity(self, current_prices: Dict[str, float] = None) -> float:
        """Calculate total portfolio equity."""
        equity = self.cash
        for symbol, pos in self.positions.items():
            if current_prices and symbol in current_prices:
                equity += pos.shares * current_prices[symbol]
            else:
                equity += pos.position_value
        return equity
    
    def record_daily_pnl(self, date: datetime, current_prices: Dict[str, float]):
        """Record daily P&L snapshot."""
        equity = self.total_equity(current_prices)
        self.daily_pnl.append({
            'date': date.strftime('%Y-%m-%d'),
            'cash': self.cash,
            'equity': equity,
            'open_positions': len(self.positions),
            'unrealized_pnl': equity - 100000.0
        })


class LiveTradingSimulator:
    """
    Simulates live trading with real-time ML risk assessment.
    
    Flow:
    1. Scan for setups on each trading day
    2. For each setup, run ML risk assessment
    3. If approved, simulate entry with position sizing
    4. Monitor positions and simulate exits
    5. Track portfolio in real-time
    """
    
    def __init__(self, strategy_name: str, start_date: datetime, end_date: datetime):
        self.strategy_name = strategy_name
        self.start_date = start_date
        self.end_date = end_date
        
        # Components
        self.screener = HistoricalParabolicScreener()
        self.risk_manager = InstitutionalRiskManager() if 'institutional' in strategy_name else None
        self.strategy = get_strategy(strategy_name)
        
        # Portfolio tracking
        self.portfolio = Portfolio()
        self.portfolio_v5 = Portfolio()  # For comparison
        
        # Statistics
        self.stats = {
            'days_scanned': 0,
            'setups_found': 0,
            'trades_taken': 0,
            'trades_blocked': 0,
            'total_pnl': 0.0,
            'wins': 0,
            'losses': 0
        }
        
        # Load symbols
        self.symbols = self._load_symbols()
        print(f"[INIT] Loaded {len(self.symbols)} symbols for scanning")
        
    def _load_symbols(self) -> List[str]:
        """Load symbol universe."""
        try:
            with open('src/backtest/extended_universe.py', 'r') as f:
                content = f.read()
                # Extract symbols from file
                import re
                match = re.search(r'EXTENDED_UNIVERSE\s*=\s*\[(.*?)\]', content, re.DOTALL)
                if match:
                    symbols_str = match.group(1)
                    symbols = [s.strip().strip('"').strip("'") for s in symbols_str.split(',')]
                    return [s for s in symbols if s]
        except:
            pass
        
        # Fallback to common micro-caps
        return pd.read_csv('reports/full_3527_backtest_results.csv')['symbol'].unique().tolist()
    
    def run_simulation(self):
        """Run the full live trading simulation."""
        print("\n" + "="*80)
        print(f"LIVE TRADING SIMULATION: {self.strategy_name.upper()}")
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Symbols: {len(self.symbols)}")
        print("="*80 + "\n")
        
        current_date = self.start_date
        
        while current_date <= self.end_date:
            # Only trade on weekdays
            if current_date.weekday() < 5:
                self._process_trading_day(current_date)
                self.stats['days_scanned'] += 1
            
            current_date += timedelta(days=1)
            
            # Progress update
            if self.stats['days_scanned'] % 30 == 0:
                self._print_progress()
        
        self._final_report()
    
    def _process_trading_day(self, date: datetime):
        """Process a single trading day."""
        # Scan for setups on this day
        setups = self._scan_day(date)
        
        if not setups:
            # Still record daily P&L for open positions
            self._update_positions_eod(date)
            return
        
        self.stats['setups_found'] += len(setups)
        
        # Process each setup
        for setup in setups:
            self._evaluate_setup(setup, date)
        
        # Update end-of-day positions
        self._update_positions_eod(date)
    
    def _scan_day(self, date: datetime) -> List[Dict]:
        """Scan for parabolic setups on a specific day."""
        setups = []
        
        # Sample symbols for efficiency (in production, scan all)
        sample_symbols = np.random.choice(self.symbols, min(100, len(self.symbols)), replace=False)
        
        for symbol in sample_symbols:
            try:
                # Quick check if symbol had movement this day
                tick_df = tick_fetcher.fetch_combined_tick_data(symbol, date, use_quotes=False)
                if tick_df.is_empty():
                    continue
                
                bar_df = tick_fetcher.aggregate_trades_to_bars(tick_df, interval_seconds=60)
                if bar_df.is_empty():
                    continue
                
                bars = bar_df.to_pandas()
                if len(bars) < 30:
                    continue
                
                # Check for parabolic move
                day_open = bars.iloc[0]['open']
                day_high = bars['high'].max()
                gain_pct = (day_high - day_open) / day_open * 100
                
                if gain_pct >= 30:  # Relaxed scanner threshold
                    setups.append({
                        'symbol': symbol,
                        'date': date,
                        'gain_pct': gain_pct,
                        'bars': bars
                    })
                    
            except Exception as e:
                continue
        
        return setups
    
    def _evaluate_setup(self, setup: Dict, date: datetime):
        """Evaluate a setup with ML risk management."""
        symbol = setup['symbol']
        bars = setup['bars']
        
        # Calculate VWAP
        bars['typical'] = (bars['high'] + bars['low'] + bars['close']) / 3
        bars['tp_v'] = bars['typical'] * bars['volume']
        bars['cum_tp_v'] = bars['tp_v'].cumsum()
        bars['cum_vol'] = bars['volume'].cumsum()
        bars['vwap'] = bars['cum_tp_v'] / bars['cum_vol']
        
        # Prepare data for risk manager
        raw_data = {
            'symbol': symbol,
            'date': date.strftime('%Y-%m-%d'),
            'bars': bars.to_dict('records')
        }
        
        if self.risk_manager:
            # Run ML risk assessment
            assessment = self.risk_manager.assess_trade(raw_data)
            
            # Log the decision
            self._log_decision(symbol, date, assessment)
            
            # Check if trade should be taken
            if assessment['recommendation'] == 'AVOID':
                self.stats['trades_blocked'] += 1
                return
            
            # Calculate position size based on Kelly
            position_size = assessment['kelly_fraction'] * 25000
        else:
            # V5 relaxed - always take trade
            assessment = {'recommendation': 'BUY', 'kelly_fraction': 1.0}
            position_size = 25000
        
        # Simulate trade
        self._simulate_trade(symbol, date, bars, position_size, assessment)
    
    def _simulate_trade(self, symbol: str, date: datetime, bars: pd.DataFrame, 
                       position_size: float, assessment: Dict):
        """Simulate a trade."""
        # Find entry in the 10-11 AM window
        entry_bars = bars[(bars.index.hour >= 10) & (bars.index.hour < 11)]
        
        if entry_bars.empty:
            return
        
        # Simple simulation: enter at first bar of window
        entry_price = entry_bars.iloc[0]['close']
        shares = int(position_size / entry_price)
        
        if shares < 1:
            return
        
        # Calculate exit (simplified VWAP target)
        vwap = bars['vwap'].iloc[-1]
        exit_price = vwap  # Target VWAP
        
        # Simulate stop loss
        stop_price = entry_price * 1.03  # 3% stop
        
        # Check if stop was hit
        if bars['high'].max() >= stop_price:
            # Stop loss hit
            pnl = (entry_price - stop_price) * shares
        else:
            # Exit at VWAP
            pnl = (entry_price - exit_price) * shares
        
        # Record trade
        trade = {
            'symbol': symbol,
            'date': date.strftime('%Y-%m-%d'),
            'entry_price': entry_price,
            'exit_price': exit_price if pnl > -position_size * 0.03 else stop_price,
            'shares': shares,
            'pnl': pnl,
            'win': 1 if pnl > 0 else 0,
            'strategy': self.strategy_name,
            'risk_score': assessment.get('risk_score', 0),
            'win_probability': assessment.get('win_probability', 0),
            'recommendation': assessment.get('recommendation', ''),
            'kelly_fraction': assessment.get('kelly_fraction', 1.0)
        }
        
        self.portfolio.closed_trades.append(trade)
        self.portfolio.cash += pnl
        
        self.stats['trades_taken'] += 1
        self.stats['total_pnl'] += pnl
        if pnl > 0:
            self.stats['wins'] += 1
        else:
            self.stats['losses'] += 1
        
        # Update risk manager with outcome
        if self.risk_manager:
            self.risk_manager.update_online({
                'predicted_win_prob': assessment['win_probability'],
                'actual_outcome': 1 if pnl > 0 else 0,
                'actual_pnl': pnl
            })
    
    def _update_positions_eod(self, date: datetime):
        """Update positions at end of day."""
        self.portfolio.record_daily_pnl(date, {})
    
    def _log_decision(self, symbol: str, date: datetime, assessment: Dict):
        """Log risk assessment decision."""
        # Only log significant decisions
        if assessment['recommendation'] in ['AVOID', 'STRONG_BUY']:
            print(f"[{date.strftime('%Y-%m-%d')}] {symbol:8s} | "
                  f"Risk: {assessment['risk_score']:.2f} | "
                  f"Win%: {assessment['win_probability']:.1%} | "
                  f"Decision: {assessment['recommendation']}")
    
    def _print_progress(self):
        """Print progress update."""
        win_rate = (self.stats['wins'] / self.stats['trades_taken'] * 100) if self.stats['trades_taken'] > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"PROGRESS: {self.stats['days_scanned']} days scanned")
        print(f"Setups Found: {self.stats['setups_found']}")
        print(f"Trades Taken: {self.stats['trades_taken']} | Blocked: {self.stats['trades_blocked']}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total P&L: ${self.stats['total_pnl']:,.2f}")
        print(f"Current Equity: ${self.portfolio.total_equity():,.2f}")
        print(f"{'='*80}\n")
    
    def _final_report(self):
        """Generate final report."""
        print("\n" + "="*80)
        print("SIMULATION COMPLETE")
        print("="*80)
        
        # Calculate final stats
        trades_df = pd.DataFrame(self.portfolio.closed_trades)
        
        if len(trades_df) > 0:
            final_equity = self.portfolio.total_equity()
            total_return = (final_equity - 100000) / 100000 * 100
            
            print(f"\nFinal Portfolio Value: ${final_equity:,.2f}")
            print(f"Total Return: {total_return:.1f}%")
            print(f"Total Trades: {len(trades_df)}")
            print(f"Win Rate: {trades_df['win'].mean()*100:.1f}%")
            print(f"Total P&L: ${trades_df['pnl'].sum():,.2f}")
            print(f"Average Trade: ${trades_df['pnl'].mean():,.2f}")
            print(f"Profit Factor: {self._calc_profit_factor(trades_df)}")
            
            # Save results
            self._save_results(trades_df)
        else:
            print("\nNo trades taken during simulation period")
    
    def _calc_profit_factor(self, df: pd.DataFrame) -> float:
        """Calculate profit factor."""
        gross_profit = df[df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(df[df['pnl'] < 0]['pnl'].sum())
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def _save_results(self, trades_df: pd.DataFrame):
        """Save simulation results."""
        output_dir = Path('reports/live_simulation')
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save trades
        trades_path = output_dir / f'{self.strategy_name}_trades.csv'
        trades_df.to_csv(trades_path, index=False)
        print(f"\nSaved trades to: {trades_path}")
        
        # Save daily P&L
        daily_df = pd.DataFrame(self.portfolio.daily_pnl)
        daily_path = output_dir / f'{self.strategy_name}_daily_pnl.csv'
        daily_df.to_csv(daily_path, index=False)
        print(f"Saved daily P&L to: {daily_path}")
        
        # Save summary
        summary = {
            'strategy': self.strategy_name,
            'start_date': self.start_date.strftime('%Y-%m-%d'),
            'end_date': self.end_date.strftime('%Y-%m-%d'),
            'final_equity': self.portfolio.total_equity(),
            'total_return_pct': (self.portfolio.total_equity() - 100000) / 100000 * 100,
            'total_trades': len(trades_df),
            'win_rate': trades_df['win'].mean() * 100,
            'total_pnl': trades_df['pnl'].sum(),
            'avg_trade': trades_df['pnl'].mean(),
            'avg_win': trades_df[trades_df['pnl'] > 0]['pnl'].mean(),
            'avg_loss': trades_df[trades_df['pnl'] < 0]['pnl'].mean(),
            'profit_factor': self._calc_profit_factor(trades_df),
            'max_drawdown': self._calc_max_drawdown(daily_df),
            'sharpe_ratio': self._calc_sharpe(daily_df)
        }
        
        summary_path = output_dir / f'{self.strategy_name}_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to: {summary_path}")
    
    def _calc_max_drawdown(self, daily_df: pd.DataFrame) -> float:
        """Calculate maximum drawdown."""
        if len(daily_df) == 0:
            return 0
        equity = daily_df['equity'].values
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        return drawdown.min() * 100
    
    def _calc_sharpe(self, daily_df: pd.DataFrame) -> float:
        """Calculate Sharpe ratio."""
        if len(daily_df) < 2:
            return 0
        returns = daily_df['equity'].pct_change().dropna()
        if returns.std() == 0:
            return 0
        return (returns.mean() / returns.std()) * np.sqrt(252)


def run_comparison():
    """Run both strategies for comparison."""
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    print("="*80)
    print("LIVE TRADING SIMULATION - STRATEGY COMPARISON")
    print("="*80)
    
    # Run V5 Relaxed
    print("\n" + "="*80)
    print("RUNNING V5 RELAXED SCANNER")
    print("="*80)
    sim_v5 = LiveTradingSimulator('v5_relaxed_scanner', start_date, end_date)
    sim_v5.run_simulation()
    
    # Run V5 Institutional
    print("\n" + "="*80)
    print("RUNNING V5 INSTITUTIONAL ML")
    print("="*80)
    sim_inst = LiveTradingSimulator('v5_institutional', start_date, end_date)
    sim_inst.run_simulation()
    
    # Generate comparison
    generate_comparison()


def generate_comparison():
    """Generate comparison between strategies."""
    print("\n" + "="*80)
    print("GENERATING COMPARISON REPORT")
    print("="*80)
    
    output_dir = Path('reports/live_simulation')
    
    # Load results
    v5_trades = pd.read_csv(output_dir / 'v5_relaxed_scanner_trades.csv')
    inst_trades = pd.read_csv(output_dir / 'v5_institutional_trades.csv')
    
    v5_summary = json.load(open(output_dir / 'v5_relaxed_scanner_summary.json'))
    inst_summary = json.load(open(output_dir / 'v5_institutional_summary.json'))
    
    # Print comparison
    print("\n" + "="*80)
    print("STRATEGY COMPARISON")
    print("="*80)
    
    print(f"\n{'Metric':<25} {'V5 Relaxed':>15} {'V5 Institutional':>20} {'Difference':>15}")
    print("-"*80)
    print(f"{'Final Equity':<25} ${v5_summary['final_equity']:>14,.0f} ${inst_summary['final_equity']:>19,.0f} ${inst_summary['final_equity']-v5_summary['final_equity']:>14,.0f}")
    print(f"{'Total Return':<25} {v5_summary['total_return_pct']:>14.1f}% {inst_summary['total_return_pct']:>19.1f}% {inst_summary['total_return_pct']-v5_summary['total_return_pct']:>14.1f}%")
    print(f"{'Total Trades':<25} {v5_summary['total_trades']:>15} {inst_summary['total_trades']:>20} {inst_summary['total_trades']-v5_summary['total_trades']:>15}")
    print(f"{'Win Rate':<25} {v5_summary['win_rate']:>14.1f}% {inst_summary['win_rate']:>19.1f}% {inst_summary['win_rate']-v5_summary['win_rate']:>14.1f}%")
    print(f"{'Total P&L':<25} ${v5_summary['total_pnl']:>14,.0f} ${inst_summary['total_pnl']:>19,.0f} ${inst_summary['total_pnl']-v5_summary['total_pnl']:>14,.0f}")
    print(f"{'Average Trade':<25} ${v5_summary['avg_trade']:>14,.0f} ${inst_summary['avg_trade']:>19,.0f} ${inst_summary['avg_trade']-v5_summary['avg_trade']:>14,.0f}")
    print(f"{'Profit Factor':<25} {v5_summary['profit_factor']:>15.2f} {inst_summary['profit_factor']:>20.2f} {inst_summary['profit_factor']-v5_summary['profit_factor']:>15.2f}")
    print(f"{'Sharpe Ratio':<25} {v5_summary['sharpe_ratio']:>15.2f} {inst_summary['sharpe_ratio']:>20.2f} {inst_summary['sharpe_ratio']-v5_summary['sharpe_ratio']:>15.2f}")
    
    # Determine winner
    if inst_summary['final_equity'] > v5_summary['final_equity']:
        winner = "V5 INSTITUTIONAL ML"
        reason = "higher final equity"
    elif inst_summary['sharpe_ratio'] > v5_summary['sharpe_ratio']:
        winner = "V5 INSTITUTIONAL ML"
        reason = "better risk-adjusted returns"
    else:
        winner = "V5 RELAXED"
        reason = "better performance"
    
    print(f"\n{'='*80}")
    print(f"WINNER: {winner}")
    print(f"Reason: {reason}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='Live trading simulation')
    parser.add_argument('--strategy', type=str, default='both',
                       choices=['v5_relaxed_scanner', 'v5_institutional', 'both'],
                       help='Strategy to simulate')
    parser.add_argument('--start-date', type=str, default='2019-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-12-31',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test on limited symbols')
    args = parser.parse_args()
    
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    if args.strategy == 'both':
        run_comparison()
    else:
        sim = LiveTradingSimulator(args.strategy, start_date, end_date)
        sim.run_simulation()


if __name__ == "__main__":
    main()
