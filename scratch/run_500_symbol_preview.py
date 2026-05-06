"""
500 SYMBOL PREVIEW - Relaxed Criteria Test
Quick validation of relaxed scanner before full run.
Expected runtime: 30-45 minutes
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


def run_500_symbol_test():
    """Run preview on first 500 symbols."""
    
    # Take first 500 symbols
    test_symbols = ALL_MICRO_CAP_SYMBOLS[:500]
    
    print("="*80)
    print("500 SYMBOL PREVIEW - Relaxed Criteria Test")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Symbols: {len(test_symbols)} (first 500 from universe)")
    print(f"Period: 2019-01-01 to 2024-12-31")
    print()
    print("RELAXED CRITERIA:")
    print("  - Min gain: 30% (was 50%)")
    print("  - Min volume: 2x average (was 3x)")
    print("  - Allow single-day parabolics")
    print("  - V5 STRICT entry (2-of-3 criteria)")
    print("="*80)
    print()
    
    # Phase 1: Scan
    print("[PHASE 1] Scanning 500 symbols with relaxed criteria...")
    screener = HistoricalParabolicScreener()
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    setups = screener.scan_for_parabolic_setups(
        symbols=test_symbols,
        start_date=start_date,
        end_date=end_date,
        min_gain_percent=30.0,
        max_gain_percent=500.0,
        min_volume_multiplier=2.0,
        use_cache=False
    )
    
    print(f"\n[SCAN RESULTS]")
    print(f"Raw parabolic setups found: {len(setups)}")
    
    # Quality filter
    quality_setups = screener.filter_quality_setups(
        setups,
        min_days_up=1,
        max_days_up=5,
        min_prior_gain=20.0,
        min_volume=50000,
        min_gain_percent=30.0
    )
    
    print(f"Quality setups: {len(quality_setups)}")
    
    # Show distribution
    if quality_setups:
        analysis = screener.analyze_setup_distribution(quality_setups)
        print(f"\n[SETUP ANALYSIS]")
        print(f"Avg gain: {analysis['avg_gain_percent']:.1f}%")
        print(f"Gain range: {analysis['gain_range'][0]:.1f}% - {analysis['gain_range'][1]:.1f}%")
        print(f"Top symbols: {', '.join([s[0] for s in analysis['top_symbols'][:5]])}")
    
    # Phase 2: Backtest
    print("\n" + "="*80)
    print("[PHASE 2] Running V5 backtest...")
    print("="*80)
    
    engine = TickBacktestEngineV5()
    results = []
    
    for i, setup in enumerate(quality_setups):
        if i % 5 == 0:
            print(f"Progress: {i}/{len(quality_setups)} setups")
        
        try:
            result = engine.run_tick_backtest(setup.symbol, setup.date, verbose=False)
            results.append({
                'symbol': setup.symbol,
                'date': setup.date.strftime('%Y-%m-%d'),
                'gain_pct': setup.gain_percent,
                'trades': result.total_trades,
                'pnl': result.total_pnl
            })
        except Exception as e:
            print(f"  Error on {setup.symbol}: {e}")
            continue
    
    # Phase 3: Results
    print("\n" + "="*80)
    print("[PHASE 3] Results & Projection")
    print("="*80)
    
    if not results:
        print("No results to analyze!")
        return
    
    df = pd.DataFrame(results)
    
    # Calculate metrics
    total_setups = len(df)
    setups_with_trades = len(df[df['trades'] > 0])
    conversion_rate = setups_with_trades / total_setups * 100 if total_setups > 0 else 0
    
    total_trades = df['trades'].sum()
    total_pnl = df['pnl'].sum()
    
    winning_trades = len(df[df['pnl'] > 0])
    losing_trades = len(df[df['pnl'] < 0])
    win_rate = winning_trades / setups_with_trades * 100 if setups_with_trades > 0 else 0
    
    print(f"\n[500 SYMBOL RESULTS]")
    print(f"Setups found:        {total_setups}")
    print(f"Trades taken:        {int(total_trades)}")
    print(f"Conversion rate:     {conversion_rate:.1f}%")
    print(f"Win rate:            {win_rate:.1f}%")
    print(f"Total P&L:           ${total_pnl:+.2f}")
    if total_trades > 0:
        print(f"Avg P&L/trade:       ${total_pnl/total_trades:.2f}")
    
    # Top winners
    print(f"\n[TOP 10 TRADES]")
    top = df.nlargest(10, 'pnl')
    for _, row in top.iterrows():
        print(f"  {row['symbol']:6} {row['date']} | ${row['pnl']:+8.2f} | {row['gain_pct']:.1f}% gap")
    
    # Projection to full universe
    print(f"\n[PROJECTION TO FULL 3,527 SYMBOLS]")
    multiplier = 3527 / 500
    print(f"Multiplier: {multiplier:.1f}x")
    print(f"Estimated setups:    {int(total_setups * multiplier)}")
    print(f"Estimated trades:    {int(total_trades * multiplier)}")
    print(f"Estimated P&L:       ${total_pnl * multiplier:+.2f}")
    
    # Comparison
    print(f"\n[COMPARISON: Original vs Relaxed (Projected)]")
    print(f"{'Metric':<20} {'Original':<15} {'Relaxed (Est.)':<15}")
    print("-" * 50)
    print(f"{'Setups':<20} {'242':<15} {str(int(total_setups * multiplier)):<15}")
    print(f"{'Trades':<20} {'40':<15} {str(int(total_trades * multiplier)):<15}")
    print(f"{'Win Rate':<20} {'80.0%':<15} {f'{win_rate:.1f}%':<15}")
    print(f"{'Total P&L':<20} {'$+53,148':<15} {f'${total_pnl * multiplier:+.0f}':<15}")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Save results
    df.to_csv("reports/500_symbol_preview.csv", index=False)
    print("\nResults saved: reports/500_symbol_preview.csv")


if __name__ == "__main__":
    run_500_symbol_test()
