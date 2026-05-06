"""
Comprehensive Live Backtest with ML Risk Management

Re-scans all symbols and runs real-time ML risk assessment on each trade.

Usage:
    python run_comprehensive_backtest.py
    
Output:
    - reports/comprehensive_v5_results.csv
    - reports/comprehensive_institutional_results.csv
    - reports/comprehensive_comparison_report.html
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import pandas as pd
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.strategies import get_strategy
from src.backtest.historical_tick_fetcher import tick_fetcher
from src.risk.ml_simple import InstitutionalRiskManager


def run_strategy_on_setups(strategy_name: str, setups_df: pd.DataFrame, 
                           use_ml_risk: bool = False) -> pd.DataFrame:
    """
    Run a strategy on all setups with optional ML risk filtering.
    
    Args:
        strategy_name: Strategy to use
        setups_df: DataFrame of setups
        use_ml_risk: Whether to use ML risk management
    
    Returns:
        DataFrame with results
    """
    print(f"\n{'='*80}")
    print(f"RUNNING: {strategy_name.upper()}")
    print(f"Setups to process: {len(setups_df)}")
    print(f"ML Risk Management: {'ENABLED' if use_ml_risk else 'DISABLED'}")
    print(f"{'='*80}\n")
    
    # Initialize strategy and risk manager
    strategy = get_strategy(strategy_name)
    risk_manager = InstitutionalRiskManager() if use_ml_risk else None
    
    results = []
    start_time = time.time()
    
    for idx, row in setups_df.iterrows():
        symbol = row['symbol']
        date_str = row['date']
        date = datetime.strptime(date_str, "%Y-%m-%d")
        
        try:
            # Fetch tick data
            tick_df = tick_fetcher.fetch_combined_tick_data(symbol, date, use_quotes=False)
            if tick_df.is_empty():
                continue
            
            # Aggregate bars
            bar_df = tick_fetcher.aggregate_trades_to_bars(tick_df, interval_seconds=60)
            if bar_df.is_empty():
                continue
            
            bars = bar_df.to_pandas()
            bars['timestamp'] = pd.to_datetime(bars['timestamp'])
            bars = bars.sort_values('timestamp')
            
            # Calculate VWAP
            bars['typical'] = (bars['high'] + bars['low'] + bars['close']) / 3
            bars['tp_v'] = bars['typical'] * bars['volume']
            bars['cum_tp_v'] = bars['tp_v'].cumsum()
            bars['cum_vol'] = bars['volume'].cumsum()
            bars['vwap'] = bars['cum_tp_v'] / bars['cum_vol']
            
            # ML Risk Assessment (if enabled)
            ml_blocked = False
            risk_score = 0.0
            win_prob = 0.0
            kelly = 1.0
            recommendation = "BUY"
            
            if use_ml_risk:
                raw_data = {
                    'symbol': symbol,
                    'date': date_str,
                    'bars': bars.to_dict('records')
                }
                
                assessment = risk_manager.assess_trade(raw_data)
                risk_score = assessment['risk_score']
                win_prob = assessment['win_probability']
                kelly = assessment['kelly_fraction']
                recommendation = assessment['recommendation']
                
                if recommendation == 'AVOID':
                    ml_blocked = True
                    # Record blocked trade
                    results.append({
                        'symbol': symbol,
                        'date': date_str,
                        'pnl': 0.0,
                        'win': 0,
                        'trades': 0,
                        'ml_blocked': True,
                        'risk_score': risk_score,
                        'win_probability': win_prob,
                        'kelly_fraction': kelly,
                        'recommendation': recommendation
                    })
                    continue
            
            # Run backtest with position sizing
            result = strategy.run_tick_backtest(symbol, date, verbose=False)
            
            # Apply Kelly sizing for institutional
            pnl = result.total_pnl
            if use_ml_risk and pnl != 0:
                pnl = pnl * kelly
            
            results.append({
                'symbol': symbol,
                'date': date_str,
                'pnl': pnl,
                'win': 1 if pnl > 0 else 0,
                'trades': result.total_trades,
                'ml_blocked': False,
                'risk_score': risk_score,
                'win_probability': win_prob,
                'kelly_fraction': kelly,
                'recommendation': recommendation
            })
            
            # Update risk manager with outcome
            if use_ml_risk and pnl != 0:
                risk_manager.update_online({
                    'predicted_win_prob': win_prob,
                    'actual_outcome': 1 if pnl > 0 else 0,
                    'actual_pnl': pnl
                })
            
        except Exception as e:
            print(f"  Error on {symbol} {date_str}: {e}")
            continue
        
        # Progress update
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            pct = (idx + 1) / len(setups_df) * 100
            eta = (elapsed / (idx + 1)) * (len(setups_df) - idx - 1)
            print(f"  Progress: {idx+1}/{len(setups_df)} ({pct:.1f}%) | ETA: {eta/60:.0f} min")
    
    return pd.DataFrame(results)


def generate_report(v5_df: pd.DataFrame, inst_df: pd.DataFrame):
    """Generate comprehensive comparison report."""
    print("\n" + "="*80)
    print("GENERATING REPORT")
    print("="*80)
    
    # Calculate statistics
    v5_trades = v5_df[v5_df['pnl'] != 0]
    inst_trades = inst_df[inst_df['pnl'] != 0]
    inst_blocked = inst_df[inst_df['ml_blocked']].shape[0]
    
    v5_stats = {
        'total_setups': len(v5_df),
        'trades_taken': len(v5_trades),
        'wins': int(v5_trades['win'].sum()),
        'win_rate': float(v5_trades['win'].mean() * 100),
        'total_pnl': float(v5_trades['pnl'].sum()),
        'avg_trade': float(v5_trades['pnl'].mean()),
        'avg_win': float(v5_trades[v5_trades['pnl'] > 0]['pnl'].mean()) if len(v5_trades[v5_trades['pnl'] > 0]) > 0 else 0,
        'avg_loss': float(v5_trades[v5_trades['pnl'] < 0]['pnl'].mean()) if len(v5_trades[v5_trades['pnl'] < 0]) > 0 else 0,
        'max_win': float(v5_trades['pnl'].max()),
        'max_loss': float(v5_trades['pnl'].min()),
    }
    
    inst_stats = {
        'total_setups': len(inst_df),
        'trades_taken': len(inst_trades),
        'trades_blocked': inst_blocked,
        'block_rate': float(inst_blocked / len(inst_df) * 100),
        'wins': int(inst_trades['win'].sum()),
        'win_rate': float(inst_trades['win'].mean() * 100),
        'total_pnl': float(inst_trades['pnl'].sum()),
        'avg_trade': float(inst_trades['pnl'].mean()),
        'avg_win': float(inst_trades[inst_trades['pnl'] > 0]['pnl'].mean()) if len(inst_trades[inst_trades['pnl'] > 0]) > 0 else 0,
        'avg_loss': float(inst_trades[inst_trades['pnl'] < 0]['pnl'].mean()) if len(inst_trades[inst_trades['pnl'] < 0]) > 0 else 0,
        'max_win': float(inst_trades['pnl'].max()),
        'max_loss': float(inst_trades['pnl'].min()),
        'avg_risk_score': float(inst_trades['risk_score'].mean()),
        'avg_win_probability': float(inst_trades['win_probability'].mean()),
    }
    
    # Print summary
    print("\n" + "="*80)
    print("V5 RELAXED SCANNER")
    print("="*80)
    print(f"Total Setups: {v5_stats['total_setups']}")
    print(f"Trades Taken: {v5_stats['trades_taken']}")
    print(f"Win Rate: {v5_stats['win_rate']:.1f}%")
    print(f"Total P&L: ${v5_stats['total_pnl']:,.2f}")
    print(f"Average Trade: ${v5_stats['avg_trade']:,.2f}")
    print(f"Average Win: ${v5_stats['avg_win']:,.2f}")
    print(f"Average Loss: ${v5_stats['avg_loss']:,.2f}")
    
    print("\n" + "="*80)
    print("V5 INSTITUTIONAL ML")
    print("="*80)
    print(f"Total Setups: {inst_stats['total_setups']}")
    print(f"Trades Taken: {inst_stats['trades_taken']}")
    print(f"Trades Blocked: {inst_stats['trades_blocked']} ({inst_stats['block_rate']:.1f}%)")
    print(f"Win Rate: {inst_stats['win_rate']:.1f}%")
    print(f"Total P&L: ${inst_stats['total_pnl']:,.2f}")
    print(f"Average Trade: ${inst_stats['avg_trade']:,.2f}")
    print(f"Average Win: ${inst_stats['avg_win']:,.2f}")
    print(f"Average Loss: ${inst_stats['avg_loss']:,.2f}")
    print(f"Avg Risk Score: {inst_stats['avg_risk_score']:.2f}")
    print(f"Avg Win Probability: {inst_stats['avg_win_probability']:.1%}")
    
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"P&L Difference: ${inst_stats['total_pnl'] - v5_stats['total_pnl']:+,.2f}")
    print(f"Win Rate Difference: {inst_stats['win_rate'] - v5_stats['win_rate']:+.1f}%")
    print(f"Trades Difference: {inst_stats['trades_taken'] - v5_stats['trades_taken']:+d}")
    
    # Calculate losses avoided
    merged = pd.merge(
        v5_df[['symbol', 'date', 'pnl']], 
        inst_df[['symbol', 'date', 'pnl', 'ml_blocked']], 
        on=['symbol', 'date'], 
        suffixes=('_v5', '_inst')
    )
    losses_avoided = merged[(merged['pnl_v5'] < 0) & (merged['ml_blocked'])]['pnl_v5'].sum()
    print(f"Losses Avoided by ML: ${abs(losses_avoided):,.2f}")
    
    # Save results
    output_dir = Path('reports/comprehensive_backtest')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    v5_df.to_csv(output_dir / 'v5_relaxed_results.csv', index=False)
    inst_df.to_csv(output_dir / 'institutional_results.csv', index=False)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'v5_relaxed': v5_stats,
        'v5_institutional': inst_stats,
        'comparison': {
            'pnl_difference': inst_stats['total_pnl'] - v5_stats['total_pnl'],
            'win_rate_difference': inst_stats['win_rate'] - v5_stats['win_rate'],
            'losses_avoided': float(abs(losses_avoided))
        }
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")


def main():
    print("="*80)
    print("COMPREHENSIVE BACKTEST WITH ML RISK MANAGEMENT")
    print("="*80)
    print("This will run both strategies on all setups with real-time ML assessment")
    print("Estimated time: 2-3 hours for full dataset")
    print("="*80)
    
    # Load setups
    setups_path = Path('reports/full_3527_setups.csv')
    if not setups_path.exists():
        print(f"[ERROR] Setups file not found: {setups_path}")
        print("Run the scanner first to generate setups")
        return
    
    setups_df = pd.read_csv(setups_path)
    print(f"\nLoaded {len(setups_df)} setups")
    
    # Ask for confirmation
    response = input(f"\nProcess all {len(setups_df)} setups? This will take ~2-3 hours. (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted.")
        return
    
    # Run V5 Relaxed
    print("\n" + "="*80)
    print("PHASE 1: V5 RELAXED SCANNER")
    print("="*80)
    v5_results = run_strategy_on_setups('v5_relaxed_scanner', setups_df, use_ml_risk=False)
    
    # Run V5 Institutional
    print("\n" + "="*80)
    print("PHASE 2: V5 INSTITUTIONAL ML")
    print("="*80)
    inst_results = run_strategy_on_setups('v5_institutional', setups_df, use_ml_risk=True)
    
    # Generate report
    generate_report(v5_results, inst_results)
    
    print("\n" + "="*80)
    print("BACKTEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
