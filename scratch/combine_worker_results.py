"""
Combine results from all parallel workers into final report.

Usage: python combine_worker_results.py
"""

import sys
import json
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    output_dir = Path('reports/parallel_backtest')
    
    if not output_dir.exists():
        print("ERROR: No worker results found!")
        print(f"Expected directory: {output_dir}")
        sys.exit(1)
    
    print("="*80)
    print("COMBINING WORKER RESULTS")
    print("="*80)
    
    # Find all worker directories
    worker_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith('worker_')])
    
    if not worker_dirs:
        print("ERROR: No worker directories found!")
        sys.exit(1)
    
    print(f"\nFound {len(worker_dirs)} worker directories")
    
    # Aggregate stats
    combined_stats = {
        'workers_completed': 0,
        'total_symbols': 0,
        'symbols_processed': 0,
        'setups_found': 0,
        'v5_trades': 0,
        'v5_pnl': 0.0,
        'ml_trades': 0,
        'ml_blocked': 0,
        'ml_pnl': 0.0,
    }
    
    all_trades = []
    
    print("\nProcessing worker results...")
    for worker_dir in worker_dirs:
        worker_id = worker_dir.name.split('_')[1]
        stats_file = worker_dir / 'stats.json'
        trades_file = worker_dir / 'trades.csv'
        
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                
            if 'end_time' in stats:
                combined_stats['workers_completed'] += 1
            
            combined_stats['total_symbols'] += stats.get('symbols_total', 0)
            combined_stats['symbols_processed'] += stats.get('symbols_done', 0)
            combined_stats['setups_found'] += stats.get('setups_found', 0)
            combined_stats['v5_trades'] += stats.get('v5_trades', 0)
            combined_stats['v5_pnl'] += stats.get('v5_pnl', 0)
            combined_stats['ml_trades'] += stats.get('ml_trades', 0)
            combined_stats['ml_blocked'] += stats.get('ml_blocked', 0)
            combined_stats['ml_pnl'] += stats.get('ml_pnl', 0)
            
            print(f"  Worker {worker_id}: {stats.get('symbols_done', 0)}/{stats.get('symbols_total', 0)} symbols, "
                  f"{stats.get('setups_found', 0)} setups, ${stats.get('v5_pnl', 0):,.0f}")
        
        if trades_file.exists():
            df = pd.read_csv(trades_file)
            all_trades.append(df)
    
    # Save combined results
    print("\nSaving combined results...")
    
    if all_trades:
        combined_df = pd.concat(all_trades, ignore_index=True)
        combined_file = output_dir / 'combined_trades.csv'
        combined_df.to_csv(combined_file, index=False)
        print(f"  Combined trades: {len(combined_df)} records -> {combined_file}")
    
    stats_file = output_dir / 'combined_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(combined_stats, f, indent=2)
    print(f"  Combined stats -> {stats_file}")
    
    # Generate report
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"\nWorkers Completed: {combined_stats['workers_completed']}/{len(worker_dirs)}")
    print(f"Symbols Processed: {combined_stats['symbols_processed']}/{combined_stats['total_symbols']}")
    print(f"Total Setups Found: {combined_stats['setups_found']}")
    print(f"\nV5 RELAXED:")
    print(f"  Trades: {combined_stats['v5_trades']}")
    print(f"  P&L: ${combined_stats['v5_pnl']:,.2f}")
    if combined_stats['v5_trades'] > 0:
        print(f"  Avg Trade: ${combined_stats['v5_pnl']/combined_stats['v5_trades']:,.2f}")
    print(f"\nV5 INSTITUTIONAL (ML):")
    print(f"  Trades Taken: {combined_stats['ml_trades']}")
    print(f"  Trades Blocked: {combined_stats['ml_blocked']}")
    print(f"  Block Rate: {combined_stats['ml_blocked']/(combined_stats['ml_trades']+combined_stats['ml_blocked'])*100:.1f}%")
    print(f"  P&L: ${combined_stats['ml_pnl']:,.2f}")
    
    # Compare
    pnl_diff = combined_stats['ml_pnl'] - combined_stats['v5_pnl']
    print(f"\nCOMPARISON:")
    print(f"  P&L Difference: ${pnl_diff:+,.2f}")
    print(f"  Winner: {'V5 Institutional' if pnl_diff > 0 else 'V5 Relaxed'}")
    
    print("\n" + "="*80)
    print("DONE!")
    print(f"Results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
