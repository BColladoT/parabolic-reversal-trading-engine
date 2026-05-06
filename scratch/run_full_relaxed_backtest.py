"""
FULL RELAXED SCAN + BACKTEST (2019-2024)
Runs comprehensive scan with relaxed criteria and full V5 backtest.
Expected runtime: 4-6 hours
"""
import sys
from pathlib import Path
from datetime import datetime
import pickle
import pandas as pd

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.backtest.historical_screener import HistoricalParabolicScreener
from src.backtest.extended_universe import ALL_MICRO_CAP_SYMBOLS
from src.backtest.tick_backtest_engine_v5 import TickBacktestEngineV5
from src.utils.logger import logger


def run_full_analysis():
    """Run complete relaxed scan and backtest."""
    
    print("="*80)
    print("FULL RELAXED SCAN + BACKTEST - 2019 to 2024")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("RELAXED DISCOVERY CRITERIA:")
    print("  - Min gain: 30% (was 50%)")
    print("  - Min volume: 2x average (was 3x)")
    print("  - Price range: $0.20 - $100")
    print("  - Allow single-day parabolics (was 2+ days)")
    print()
    print("STRICT ENTRY CRITERIA (V5):")
    print("  - 2-of-3 criteria required")
    print("  - VWAP extension > 15%")
    print("  - Volume < 70% of peak")
    print("  - Within 7% of HOD")
    print()
    print(f"Universe: {len(ALL_MICRO_CAP_SYMBOLS):,} symbols")
    print(f"Period: 2019-01-01 to 2024-12-31 (~6 years)")
    print("="*80)
    print()
    
    # Phase 1: Scan
    print("[PHASE 1] Scanning for parabolic setups with relaxed criteria...")
    print()
    
    screener = HistoricalParabolicScreener()
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    setups = screener.scan_for_parabolic_setups(
        symbols=ALL_MICRO_CAP_SYMBOLS,
        start_date=start_date,
        end_date=end_date,
        min_gain_percent=30.0,       # RELAXED: 30% (was 50%)
        max_gain_percent=500.0,
        min_volume_multiplier=2.0,   # RELAXED: 2x (was 3x)
        use_cache=False              # Force fresh scan
    )
    
    print(f"\n[SCAN COMPLETE]")
    print(f"Total parabolic setups found: {len(setups)}")
    
    # Phase 2: Quality Filter
    print("\n[PHASE 2] Applying quality filters...")
    
    quality_setups = screener.filter_quality_setups(
        setups,
        min_days_up=1,               # RELAXED: Allow single-day
        max_days_up=5,
        min_prior_gain=20.0,         # RELAXED: 20% (was 30%)
        min_volume=50000,            # RELAXED: 50K (was 100K)
        min_gain_percent=30.0        # RELAXED: 30% (was 50%)
    )
    
    print(f"Quality setups: {len(quality_setups)}")
    
    # Save setups
    cache_path = Path("data/cache/setups/setups_relaxed_full_2019_2024.pkl")
    with open(cache_path, 'wb') as f:
        pickle.dump(quality_setups, f)
    print(f"Saved to: {cache_path}")
    
    # Export CSV
    csv_path = screener.export_setups_for_backtest(
        quality_setups, 
        "reports/relaxed_setups_full.csv"
    )
    
    # Phase 3: Backtest
    print("\n" + "="*80)
    print("[PHASE 3] Running V5 backtest on all setups...")
    print("="*80)
    print()
    
    engine = TickBacktestEngineV5()
    results = []
    
    for i, setup in enumerate(quality_setups):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(quality_setups)} setups tested")
        
        try:
            result = engine.run_tick_backtest(
                setup.symbol, 
                setup.date, 
                verbose=False
            )
            
            results.append({
                'symbol': setup.symbol,
                'date': setup.date.strftime('%Y-%m-%d'),
                'gain_pct': setup.gain_percent,
                'trades': result.total_trades,
                'pnl': result.total_pnl,
                'win_rate': result.win_rate * 100 if result.total_trades > 0 else 0
            })
            
        except Exception as e:
            print(f"  Error on {setup.symbol} {setup.date}: {e}")
            continue
    
    # Phase 4: Analysis
    print("\n" + "="*80)
    print("[PHASE 4] Analysis & Reporting")
    print("="*80)
    
    df = pd.DataFrame(results)
    
    # Calculate metrics
    total_setups = len(df)
    setups_with_trades = len(df[df['trades'] > 0])
    conversion_rate = setups_with_trades / total_setups * 100 if total_setups > 0 else 0
    
    total_trades = df['trades'].sum()
    total_pnl = df['pnl'].sum()
    
    winning_trades = len(df[df['pnl'] > 0])
    losing_trades = len(df[df['pnl'] <= 0])
    win_rate = winning_trades / setups_with_trades * 100 if setups_with_trades > 0 else 0
    
    avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0
    avg_pnl_per_setup = total_pnl / total_setups if total_setups > 0 else 0
    
    print(f"\n[RESULTS SUMMARY]")
    print(f"Total Setups Scanned:     {total_setups}")
    print(f"Setups with Trades:       {setups_with_trades}")
    print(f"Conversion Rate:          {conversion_rate:.1f}%")
    print(f"")
    print(f"Total Trades Executed:    {total_trades}")
    print(f"Winning Trades:           {winning_trades}")
    print(f"Losing Trades:            {losing_trades}")
    print(f"Win Rate:                 {win_rate:.1f}%")
    print(f"")
    print(f"Total P&L:                ${total_pnl:+.2f}")
    print(f"Avg P&L per Trade:        ${avg_pnl_per_trade:.2f}")
    print(f"Avg P&L per Setup:        ${avg_pnl_per_setup:.2f}")
    
    # Top winners
    print(f"\n[TOP 15 WINNING TRADES]")
    top_winners = df.nlargest(15, 'pnl')
    for idx, row in top_winners.iterrows():
        print(f"  {row['symbol']:6} {row['date']} | ${row['pnl']:+8.2f} | {row['gain_pct']:.1f}% gap")
    
    # Top losers
    print(f"\n[TOP 10 LOSING TRADES]")
    top_losers = df.nsmallest(10, 'pnl')
    for idx, row in top_losers.iterrows():
        print(f"  {row['symbol']:6} {row['date']} | ${row['pnl']:+8.2f} | {row['gain_pct']:.1f}% gap")
    
    # Monthly distribution
    df['date'] = pd.to_datetime(df['date'])
    df['year_month'] = df['date'].dt.to_period('M')
    monthly = df.groupby('year_month').agg({
        'trades': 'sum',
        'pnl': 'sum'
    }).reset_index()
    
    print(f"\n[MONTHLY ACTIVITY (Sample)]")
    print(f"{'Month':10} {'Trades':>8} {'P&L':>12}")
    print("-" * 35)
    for idx, row in monthly.head(10).iterrows():
        print(f"{str(row['year_month']):10} {row['trades']:>8} ${row['pnl']:>+10.2f}")
    
    # Save results
    results_df = df[['symbol', 'date', 'gain_pct', 'trades', 'pnl', 'win_rate']]
    results_path = "reports/relaxed_backtest_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved: {results_path}")
    
    # Comparison with original
    print("\n" + "="*80)
    print("[COMPARISON: Original vs Relaxed]")
    print("="*80)
    print(f"{'Metric':<25} {'Original (50%)':<20} {'Relaxed (30%)':<20}")
    print("-" * 65)
    print(f"{'Setups Found':<25} {'242':<20} {str(total_setups):<20}")
    print(f"{'Trades Taken':<25} {'40':<20} {str(int(total_trades)):<20}")
    print(f"{'Win Rate':<25} {'80.0%':<20} {f'{win_rate:.1f}%':<20}")
    print(f"{'Total P&L':<25} {'$+53,148':<20} {f'${total_pnl:+.0f}':<20}")
    print(f"{'Avg P&L/Trade':<25} {'$+1,329':<20} {f'${avg_pnl_per_trade:.0f}':<20}")
    print(f"{'Trades/Year':<25} {'~7':<20} {f'~{int(total_trades/6)}':<20}")
    
    print("\n" + "="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == "__main__":
    run_full_analysis()
