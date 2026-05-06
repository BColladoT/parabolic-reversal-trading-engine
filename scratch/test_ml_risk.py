"""
Test ML Risk Management Strategy vs Base V5

Compares performance of V5 with ML risk filtering against base V5
on the losing trades to see if we would have filtered them out.
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.strategies.v5_ml_risk import TickBacktestEngineV5_MLRisk
from src.strategies.v5_strict import TickBacktestEngineV5

# Load the worst losing trades
losses_df = pd.read_csv('reports/losing_trades_analysis.csv')
worst_losses = losses_df.nsmallest(20, 'pnl')

print("="*80)
print("TESTING ML RISK MANAGEMENT ON WORST LOSING TRADES")
print("="*80)

# Test with ML-enhanced strategy
print("\n[1] Testing V5 + ML Risk Filter on worst losses...")
engine_ml = TickBacktestEngineV5_MLRisk()

ml_results = []
for idx, row in worst_losses.iterrows():
    symbol = row['symbol']
    date_str = row['date']
    original_pnl = row['pnl']
    date = datetime.strptime(date_str, "%Y-%m-%d")
    
    print(f"\n[{idx+1}] Testing {symbol} on {date_str} (Original Loss: ${original_pnl:,.2f})")
    
    try:
        result = engine_ml.run_tick_backtest(symbol, date, verbose=True)
        ml_results.append({
            'symbol': symbol,
            'date': date_str,
            'original_pnl': original_pnl,
            'ml_pnl': result.total_pnl,
            'trade_taken': result.total_pnl != 0 or result.total_trades > 0
        })
    except Exception as e:
        print(f"  [ERROR] {e}")
        ml_results.append({
            'symbol': symbol,
            'date': date_str,
            'original_pnl': original_pnl,
            'ml_pnl': 0,
            'trade_taken': False,
            'error': str(e)
        })

# Summary
print("\n" + "="*80)
print("ML RISK FILTER RESULTS SUMMARY")
print("="*80)

ml_df = pd.DataFrame(ml_results)
original_total_loss = ml_df['original_pnl'].sum()
ml_total_pnl = ml_df['ml_pnl'].sum()
trades_filtered = (~ml_df['trade_taken']).sum()

print(f"\nOriginal total losses on these 20 trades: ${original_total_loss:,.2f}")
print(f"ML filter total P&L on these 20 trades: ${ml_total_pnl:,.2f}")
print(f"Trades filtered out: {trades_filtered}/20")

if trades_filtered > 0:
    filtered_pnl = ml_df[~ml_df['trade_taken']]['original_pnl'].sum()
    print(f"Losses avoided by filtering: ${filtered_pnl:,.2f}")

# Risk manager stats
risk_report = engine_ml.get_risk_report()
print(f"\n[ML Risk Manager Stats]")
print(f"  Total trades filtered: {risk_report['trades_filtered']}")
print(f"  Estimated P&L saved: ${risk_report['estimated_pnl_saved']:,.2f}")
print(f"  Average risk score: {risk_report['avg_risk_score']:.3f}")

print("\n" + "="*80)
print("DETAILED COMPARISON")
print("="*80)
for _, row in ml_df.iterrows():
    status = "FILTERED" if not row['trade_taken'] else "TAKEN"
    diff = row['ml_pnl'] - row['original_pnl']
    print(f"{row['symbol']:6s} {row['date']}: Original=${row['original_pnl']:>10,.2f} | "
          f"ML={row['ml_pnl']:>10,.2f} | Diff=${diff:>10,.2f} | {status}")
