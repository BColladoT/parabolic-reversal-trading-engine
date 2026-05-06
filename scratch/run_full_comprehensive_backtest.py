"""
FULL COMPREHENSIVE BACKTEST - All 3,527 Symbols
Relaxed Scanner (30% gain) + V5 Strict Entry
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


def run_full_backtest():
    """Run complete backtest on all symbols."""
    
    start_time = datetime.now()
    
    print("="*80)
    print("FULL COMPREHENSIVE BACKTEST - ALL SYMBOLS")
    print("="*80)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Universe: {len(ALL_MICRO_CAP_SYMBOLS):,} symbols")
    print(f"Period: 2019-01-01 to 2024-12-31 (6 years)")
    print()
    print("CONFIGURATION:")
    print("  Scanner: Relaxed (30% gain, 2x volume, single-day allowed)")
    print("  Entry: V5 Strict (2-of-3 criteria, VWAP>15%, Vol<70%, Prox>93%)")
    print("="*80)
    print()
    
    # Phase 1: Scan all symbols
    print("[PHASE 1] Scanning all symbols for parabolic setups...")
    print()
    
    screener = HistoricalParabolicScreener()
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    all_setups = screener.scan_for_parabolic_setups(
        symbols=ALL_MICRO_CAP_SYMBOLS,
        start_date=start_date,
        end_date=end_date,
        min_gain_percent=30.0,      # Relaxed: 30%
        max_gain_percent=500.0,
        min_volume_multiplier=2.0,  # Relaxed: 2x
        use_cache=False             # Force fresh scan
    )
    
    scan_complete = datetime.now()
    scan_duration = (scan_complete - start_time).total_seconds() / 60
    
    print(f"\n[SCAN COMPLETE]")
    print(f"Duration: {scan_duration:.1f} minutes")
    print(f"Raw setups found: {len(all_setups)}")
    
    # Quality filter
    quality_setups = screener.filter_quality_setups(
        all_setups,
        min_days_up=1,
        max_days_up=5,
        min_prior_gain=20.0,
        min_volume=50000,
        min_gain_percent=30.0
    )
    
    print(f"Quality setups: {len(quality_setups)}")
    
    # Save setups
    cache_path = Path("data/cache/setups/full_3527_setups.pkl")
    with open(cache_path, 'wb') as f:
        pickle.dump(quality_setups, f)
    print(f"Saved to: {cache_path}")
    
    # Export CSV
    csv_path = screener.export_setups_for_backtest(
        quality_setups, 
        "reports/full_3527_setups.csv"
    )
    
    # Phase 2: Backtest all setups
    print("\n" + "="*80)
    print(f"[PHASE 2] Running V5 backtest on {len(quality_setups)} setups...")
    print("="*80)
    print()
    
    engine = TickBacktestEngineV5()
    results = []
    errors = []
    
    for i, setup in enumerate(quality_setups):
        if i % 10 == 0:
            progress = i / len(quality_setups) * 100
            elapsed = (datetime.now() - scan_complete).total_seconds() / 60
            print(f"Progress: {i}/{len(quality_setups)} ({progress:.1f}%) | Elapsed: {elapsed:.1f} min")
        
        try:
            result = engine.run_tick_backtest(setup.symbol, setup.date, verbose=False)
            results.append({
                'symbol': setup.symbol,
                'date': setup.date.strftime('%Y-%m-%d'),
                'gain_pct': setup.gain_percent,
                'days_up': setup.days_up,
                'volume': setup.day_volume,
                'trades': result.total_trades,
                'pnl': result.total_pnl,
                'win': 1 if result.total_pnl > 0 else 0,
                'loss': 1 if result.total_pnl <= 0 else 0
            })
        except Exception as e:
            errors.append({'symbol': setup.symbol, 'date': setup.date, 'error': str(e)})
            continue
    
    backtest_complete = datetime.now()
    backtest_duration = (backtest_complete - scan_complete).total_seconds() / 60
    
    print(f"\n[BACKTEST COMPLETE]")
    print(f"Duration: {backtest_duration:.1f} minutes")
    print(f"Setups processed: {len(results)}")
    print(f"Errors: {len(errors)}")
    
    # Phase 3: Analysis
    print("\n" + "="*80)
    print("[PHASE 3] Generating comprehensive report...")
    print("="*80)
    
    df = pd.DataFrame(results)
    
    # Calculate all metrics
    total_setups = len(df)
    setups_with_trades = len(df[df['trades'] > 0])
    conversion_rate = setups_with_trades / total_setups * 100 if total_setups > 0 else 0
    
    total_trades = df['trades'].sum()
    total_pnl = df['pnl'].sum()
    winning_trades = df['win'].sum()
    losing_trades = df['loss'].sum()
    win_rate = winning_trades / setups_with_trades * 100 if setups_with_trades > 0 else 0
    
    avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0
    avg_pnl_per_setup = total_pnl / total_setups if total_setups > 0 else 0
    
    # Save results
    results_path = "reports/full_3527_backtest_results.csv"
    df.to_csv(results_path, index=False)
    
    # Save errors if any
    if errors:
        pd.DataFrame(errors).to_csv("reports/full_3527_errors.csv", index=False)
    
    # Print final report
    total_duration = (datetime.now() - start_time).total_seconds() / 60
    
    print("\n" + "="*80)
    print("FINAL RESULTS - FULL 3,527 SYMBOL BACKTEST")
    print("="*80)
    print(f"Total runtime: {total_duration:.1f} minutes")
    print()
    
    print("[SETUP STATISTICS]")
    print(f"Total symbols scanned:    {len(ALL_MICRO_CAP_SYMBOLS):,}")
    print(f"Total setups found:       {total_setups}")
    print(f"Setups with trades:       {setups_with_trades}")
    print(f"Conversion rate:          {conversion_rate:.1f}%")
    print()
    
    print("[TRADE STATISTICS]")
    print(f"Total trades executed:    {int(total_trades)}")
    print(f"Winning trades:           {winning_trades}")
    print(f"Losing trades:            {losing_trades}")
    print(f"Win rate:                 {win_rate:.1f}%")
    print()
    
    print("[P&L STATISTICS]")
    print(f"Total P&L:                ${total_pnl:+.2f}")
    print(f"Avg P&L per trade:        ${avg_pnl_per_trade:.2f}")
    print(f"Avg P&L per setup:        ${avg_pnl_per_setup:.2f}")
    print()
    
    # Top winners
    print("[TOP 20 WINNING TRADES]")
    top = df.nlargest(20, 'pnl')
    for i, (_, row) in enumerate(top.iterrows(), 1):
        print(f"{i:2}. {row['symbol']:6} {row['date']} | ${row['pnl']:+9.2f} | {row['gain_pct']:5.1f}% gap")
    
    # Top losers
    print("\n[TOP 10 LOSING TRADES]")
    bottom = df.nsmallest(10, 'pnl')
    for i, (_, row) in enumerate(bottom.iterrows(), 1):
        print(f"{i:2}. {row['symbol']:6} {row['date']} | ${row['pnl']:+9.2f} | {row['gain_pct']:5.1f}% gap")
    
    # Comparison
    print("\n" + "="*80)
    print("COMPARISON: Original vs Full Relaxed")
    print("="*80)
    orig_pnl = 53148.33
    orig_trades = 40
    print(f"{'Metric':<25} {'Original':<18} {'Full Relaxed':<18}")
    print("-" * 62)
    print(f"{'Symbols Scanned':<25} {'1,115':<18} {'3,527':<18}")
    print(f"{'Setups Found':<25} {'242':<18} {str(total_setups):<18}")
    print(f"{'Trades Taken':<25} {str(orig_trades):<18} {str(int(total_trades)):<18}")
    print(f"{'Win Rate':<25} {'80.0%':<18} {f'{win_rate:.1f}%':<18}")
    print(f"{'Total P&L':<25} {f'${orig_pnl:,.0f}':<18} {f'${total_pnl:,.0f}':<18}")
    print(f"{'Trades/Year':<25} {'~7':<18} {f'~{int(total_trades/6)}':<18}")
    
    # Yearly breakdown
    print("\n[YEARLY BREAKDOWN]")
    df['year'] = pd.to_datetime(df['date']).dt.year
    yearly = df.groupby('year').agg({
        'trades': 'sum',
        'pnl': 'sum'
    }).reset_index()
    for _, row in yearly.iterrows():
        print(f"  {int(row['year'])}: {int(row['trades'])} trades | ${row['pnl']:+.2f}")
    
    print("\n" + "="*80)
    print("FILES SAVED:")
    print(f"  - {results_path}")
    print(f"  - {csv_path}")
    print(f"  - {cache_path}")
    if errors:
        print(f"  - reports/full_3527_errors.csv ({len(errors)} errors)")
    print("="*80)
    
    return df


if __name__ == "__main__":
    run_full_backtest()
