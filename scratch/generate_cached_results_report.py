"""
Generate report and charts from cached parallel backtest results.
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

print("="*80)
print("CACHED PARALLEL BACKTEST RESULTS")
print("="*80)

# Load data
output_dir = Path('reports/cached_parallel_backtest')
df = pd.read_csv(output_dir / 'combined_trades.csv')
stats = json.load(open(output_dir / 'combined_stats.json'))

print(f"\n[SUMMARY]")
print(f"Symbols Processed: {stats['symbols_processed']}")
print(f"Setups Found: {stats['setups_found']}")

# V5 Analysis
v5_df = df[df['strategy'] == 'v5_relaxed']
v5_trades = v5_df[v5_df['pnl'] != 0]

print(f"\n[V5 RELAXED]")
print(f"  Trades: {len(v5_trades)}")
if len(v5_trades) > 0:
    v5_win_rate = (v5_trades['win'] == 1).mean() * 100
    v5_pnl = v5_trades['pnl'].sum()
    v5_avg = v5_trades['pnl'].mean()
    print(f"  Win Rate: {v5_win_rate:.1f}%")
    print(f"  Total P&L: ${v5_pnl:,.2f}")
    print(f"  Avg Trade: ${v5_avg:,.2f}")
    print(f"  Max Win: ${v5_trades['pnl'].max():,.2f}")
    print(f"  Max Loss: ${v5_trades['pnl'].min():,.2f}")

# ML Analysis
inst_df = df[df['strategy'] == 'v5_institutional']
inst_taken = inst_df[inst_df['ml_blocked'] == False]
inst_blocked = inst_df[inst_df['ml_blocked'] == True]

print(f"\n[V5 INSTITUTIONAL (ML)]")
print(f"  Trades Taken: {len(inst_taken)}")
print(f"  Trades Blocked: {len(inst_blocked)}")
if len(inst_taken) + len(inst_blocked) > 0:
    block_rate = len(inst_blocked) / (len(inst_taken) + len(inst_blocked)) * 100
    print(f"  Block Rate: {block_rate:.1f}%")

if len(inst_taken) > 0:
    ml_win_rate = (inst_taken['win'] == 1).mean() * 100
    ml_pnl = inst_taken['pnl'].sum()
    ml_avg = inst_taken['pnl'].mean()
    print(f"  Win Rate: {ml_win_rate:.1f}%")
    print(f"  Total P&L: ${ml_pnl:,.2f}")
    print(f"  Avg Trade: ${ml_avg:,.2f}")
    print(f"  Max Win: ${inst_taken['pnl'].max():,.2f}")
    print(f"  Max Loss: ${inst_taken['pnl'].min():,.2f}")

# Comparison
print(f"\n[COMPARISON]")
if len(v5_trades) > 0 and len(inst_taken) > 0:
    pnl_diff = ml_pnl - v5_pnl
    print(f"  P&L Difference: ${pnl_diff:+,.2f}")
    print(f"  Winner: {'V5 Institutional' if pnl_diff > 0 else 'V5 Relaxed'}")

# Generate HTML Report
print("\n[Generating HTML report...]")

html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Cached Parallel Backtest Results</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }}
        h1 {{ color: #00d4ff; }}
        h2 {{ color: #00d4ff; border-bottom: 2px solid #00d4ff; padding-bottom: 10px; }}
        .metric-card {{ background: #16213e; padding: 20px; margin: 10px; border-radius: 10px; display: inline-block; min-width: 200px; }}
        .metric-value {{ font-size: 32px; font-weight: bold; color: #00d4ff; }}
        .metric-label {{ font-size: 14px; color: #888; }}
        .positive {{ color: #00ff88; }}
        .negative {{ color: #ff4757; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #333; }}
        th {{ background: #16213e; color: #00d4ff; }}
        .chart-container {{ background: #16213e; padding: 20px; margin: 20px 0; border-radius: 10px; }}
    </style>
</head>
<body>
    <h1>Cached Parallel Backtest Results</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Summary</h2>
    <div class="metric-card">
        <div class="metric-value">{stats['symbols_processed']}</div>
        <div class="metric-label">Symbols Processed</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{stats['setups_found']}</div>
        <div class="metric-label">Setups Found</div>
    </div>
"""

if len(v5_trades) > 0:
    html += f"""
    <h2>V5 Relaxed Performance</h2>
    <div class="metric-card">
        <div class="metric-value">{len(v5_trades)}</div>
        <div class="metric-label">Trades</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{v5_win_rate:.1f}%</div>
        <div class="metric-label">Win Rate</div>
    </div>
    <div class="metric-card">
        <div class="metric-value {'positive' if v5_pnl > 0 else 'negative'}">${v5_pnl:,.0f}</div>
        <div class="metric-label">Total P&L</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">${v5_avg:,.0f}</div>
        <div class="metric-label">Avg Trade</div>
    </div>
"""

if len(inst_taken) > 0:
    html += f"""
    <h2>V5 Institutional (ML) Performance</h2>
    <div class="metric-card">
        <div class="metric-value">{len(inst_taken)}</div>
        <div class="metric-label">Trades Taken</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{len(inst_blocked)}</div>
        <div class="metric-label">Trades Blocked</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{block_rate:.1f}%</div>
        <div class="metric-label">Block Rate</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{ml_win_rate:.1f}%</div>
        <div class="metric-label">Win Rate</div>
    </div>
    <div class="metric-card">
        <div class="metric-value {'positive' if ml_pnl > 0 else 'negative'}">${ml_pnl:,.0f}</div>
        <div class="metric-label">Total P&L</div>
    </div>
"""

if len(v5_trades) > 0 and len(inst_taken) > 0:
    html += f"""
    <h2>Comparison</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>V5 Relaxed</th>
            <th>V5 Institutional</th>
            <th>Difference</th>
        </tr>
        <tr>
            <td>Trades</td>
            <td>{len(v5_trades)}</td>
            <td>{len(inst_taken)}</td>
            <td>{len(inst_taken) - len(v5_trades):+d}</td>
        </tr>
        <tr>
            <td>Win Rate</td>
            <td>{v5_win_rate:.1f}%</td>
            <td>{ml_win_rate:.1f}%</td>
            <td class="{'positive' if ml_win_rate > v5_win_rate else 'negative'}">{ml_win_rate - v5_win_rate:+.1f}%</td>
        </tr>
        <tr>
            <td>Total P&L</td>
            <td>${v5_pnl:,.0f}</td>
            <td>${ml_pnl:,.0f}</td>
            <td class="{'positive' if pnl_diff > 0 else 'negative'}">${pnl_diff:+,.0f}</td>
        </tr>
        <tr>
            <td>Avg Trade</td>
            <td>${v5_avg:,.0f}</td>
            <td>${ml_avg:,.0f}</td>
            <td class="{'positive' if ml_avg > v5_avg else 'negative'}">${ml_avg - v5_avg:+,.0f}</td>
        </tr>
    </table>
    
    <h2>Winner: {'V5 Institutional' if pnl_diff > 0 else 'V5 Relaxed'}</h2>
    <p style="font-size: 24px; color: {'#00ff88' if pnl_diff > 0 else '#ff4757'};">
        {'ML Risk filtering improved performance by' if pnl_diff > 0 else 'V5 Relaxed outperformed ML by'} ${abs(pnl_diff):,.0f}
    </p>
"""

# Add charts
html += """
    <h2>Charts</h2>
    <div class="chart-container">
        <div id="pnlDist"></div>
    </div>
    <div class="chart-container">
        <div id="comparisonChart"></div>
    </div>
"""

# Add JavaScript for charts
if len(v5_trades) > 0 and len(inst_taken) > 0:
    v5_pnls = v5_trades['pnl'].tolist()
    ml_pnls = inst_taken['pnl'].tolist()
    
    html += f"""
    <script>
        // P&L Distribution
        var trace1 = {{
            x: {v5_pnls},
            name: 'V5 Relaxed',
            opacity: 0.75,
            type: 'histogram',
            marker: {{color: '#00d4ff'}}
        }};
        var trace2 = {{
            x: {ml_pnls},
            name: 'V5 Institutional',
            opacity: 0.75,
            type: 'histogram',
            marker: {{color: '#ff4757'}}
        }};
        Plotly.newPlot('pnlDist', [trace1, trace2], {{
            title: 'Trade P&L Distribution',
            barmode: 'overlay',
            paper_bgcolor: '#16213e',
            plot_bgcolor: '#16213e',
            font: {{color: '#eee'}},
            xaxis: {{title: 'P&L ($)'}},
            yaxis: {{title: 'Frequency'}}
        }});
        
        // Comparison Bar Chart
        Plotly.newPlot('comparisonChart', [
            {{x: ['Total P&L', 'Win Rate (x1000)', 'Avg Trade'], 
              y: [{v5_pnl}, {v5_win_rate*10}, {v5_avg}], 
              name: 'V5 Relaxed', type: 'bar', marker: {{color: '#00d4ff'}}}},
            {{x: ['Total P&L', 'Win Rate (x1000)', 'Avg Trade'], 
              y: [{ml_pnl}, {ml_win_rate*10}, {ml_avg}], 
              name: 'V5 Institutional', type: 'bar', marker: {{color: '#ff4757'}}}}
        ], {{
            title: 'Strategy Comparison',
            barmode: 'group',
            paper_bgcolor: '#16213e',
            plot_bgcolor: '#16213e',
            font: {{color: '#eee'}}
        }});
    </script>
"""

html += """
</body>
</html>
"""

# Save report
report_path = output_dir / 'results_report.html'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\n[OK] Report saved to: {report_path}")
print("="*80)
