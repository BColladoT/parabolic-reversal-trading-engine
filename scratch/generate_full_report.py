"""
Generate comprehensive charts from completed backtest results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Create output directory
output_dir = Path('reports/analysis_charts')
output_dir.mkdir(exist_ok=True, parents=True)

print("="*80)
print("GENERATING CHARTS FROM COMPLETED BACKTEST RESULTS")
print("="*80)

# Load data
print("\n[1/5] Loading backtest data...")
relaxed_df = pd.read_csv('reports/relaxed_909_backtest.csv')
comparison_df = pd.read_csv('reports/comparison_backtest_results.csv')

with open('reports/comparison_summary.json', 'r') as f:
    comparison_summary = json.load(f)

print(f"  Relaxed backtest: {len(relaxed_df)} setups")
print(f"  Comparison data: {len(comparison_df)} records")

# Calculate metrics
print("\n[2/5] Calculating performance metrics...")

# Overall statistics
stats = {
    'total_setups': len(relaxed_df),
    'unique_symbols': relaxed_df['symbol'].nunique(),
    'date_range': f"{relaxed_df['date'].min()} to {relaxed_df['date'].max()}",
}

# V5 Relaxed metrics
v5_trades = relaxed_df[relaxed_df['trades'] > 0]
if len(v5_trades) > 0:
    stats['v5'] = {
        'trades_taken': len(v5_trades),
        'total_pnl': v5_trades['pnl'].sum(),
        'win_rate': (v5_trades['pnl'] > 0).mean() * 100,
        'avg_trade': v5_trades['pnl'].mean(),
        'max_win': v5_trades['pnl'].max(),
        'max_loss': v5_trades['pnl'].min(),
    }

# ML Comparison metrics
if 'strategy' in comparison_df.columns:
    v5_comp = comparison_df[comparison_df['strategy'] == 'v5_relaxed']
    inst_comp = comparison_df[comparison_df['strategy'] == 'v5_institutional']
    
    stats['comparison'] = {
        'v5_trades': len(v5_comp),
        'v5_pnl': v5_comp['pnl'].sum(),
        'v5_win_rate': (v5_comp['pnl'] > 0).mean() * 100 if len(v5_comp) > 0 else 0,
        'inst_trades': len(inst_comp),
        'inst_pnl': inst_comp['pnl'].sum(),
        'inst_win_rate': (inst_comp['pnl'] > 0).mean() * 100 if len(inst_comp) > 0 else 0,
    }

print(f"  V5 Relaxed: {stats['v5']['trades_taken']} trades, ${stats['v5']['total_pnl']:,.0f} P&L")

# Generate HTML report
print("\n[3/5] Generating HTML report...")

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Parabolic Reversal Backtest Results</title>
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
    <h1>📊 Parabolic Reversal Backtest Results</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>📈 Overall Statistics</h2>
    <div class="metric-card">
        <div class="metric-value">{stats['total_setups']:,}</div>
        <div class="metric-label">Total Setups Found</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{stats['unique_symbols']:,}</div>
        <div class="metric-label">Unique Symbols</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{stats['v5']['trades_taken']}</div>
        <div class="metric-label">Trades Taken (V5)</div>
    </div>
    
    <h2>💰 V5 Relaxed Performance</h2>
    <div class="metric-card">
        <div class="metric-value {'positive' if stats['v5']['total_pnl'] > 0 else 'negative'}">${stats['v5']['total_pnl']:,.0f}</div>
        <div class="metric-label">Total P&L</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{stats['v5']['win_rate']:.1f}%</div>
        <div class="metric-label">Win Rate</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">${stats['v5']['avg_trade']:,.0f}</div>
        <div class="metric-label">Avg Trade</div>
    </div>
    <div class="metric-card">
        <div class="metric-value positive">${stats['v5']['max_win']:,.0f}</div>
        <div class="metric-label">Max Win</div>
    </div>
    <div class="metric-card">
        <div class="metric-value negative">${stats['v5']['max_loss']:,.0f}</div>
        <div class="metric-label">Max Loss</div>
    </div>
"""

# Add comparison section if available
if 'comparison' in stats:
    html_content += f"""
    <h2>🤖 V5 vs V5 Institutional (ML Risk)</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>V5 Relaxed</th>
            <th>V5 Institutional</th>
            <th>Difference</th>
        </tr>
        <tr>
            <td>Trades Taken</td>
            <td>{stats['comparison']['v5_trades']}</td>
            <td>{stats['comparison']['inst_trades']}</td>
            <td>{stats['comparison']['inst_trades'] - stats['comparison']['v5_trades']:+d}</td>
        </tr>
        <tr>
            <td>Win Rate</td>
            <td>{stats['comparison']['v5_win_rate']:.1f}%</td>
            <td>{stats['comparison']['inst_win_rate']:.1f}%</td>
            <td class="{'positive' if stats['comparison']['inst_win_rate'] > stats['comparison']['v5_win_rate'] else 'negative'}">{stats['comparison']['inst_win_rate'] - stats['comparison']['v5_win_rate']:+.1f}%</td>
        </tr>
        <tr>
            <td>Total P&L</td>
            <td>${stats['comparison']['v5_pnl']:,.0f}</td>
            <td>${stats['comparison']['inst_pnl']:,.0f}</td>
            <td class="{'positive' if stats['comparison']['inst_pnl'] > stats['comparison']['v5_pnl'] else 'negative'}">${stats['comparison']['inst_pnl'] - stats['comparison']['v5_pnl']:+,.0f}</td>
        </tr>
    </table>
"""

# Add charts
html_content += """
    <h2>📊 Performance Charts</h2>
    <div class="chart-container">
        <div id="pnlChart"></div>
    </div>
    <div class="chart-container">
        <div id="equityChart"></div>
    </div>
    <div class="chart-container">
        <div id="distributionChart"></div>
    </div>
"""

# Add JSON summary
html_content += f"""
    <h2>📋 Summary JSON</h2>
    <pre style="background: #16213e; padding: 20px; border-radius: 10px; overflow-x: auto;">{json.dumps(stats, indent=2)}</pre>
    
    <script>
        // P&L by Symbol
        var pnlData = {{
            x: {relaxed_df.groupby('symbol')['pnl'].sum().sort_values(ascending=False).head(20).index.tolist()},
            y: {relaxed_df.groupby('symbol')['pnl'].sum().sort_values(ascending=False).head(20).values.tolist()},
            type: 'bar',
            marker: {{
                color: {relaxed_df.groupby('symbol')['pnl'].sum().sort_values(ascending=False).head(20).values.tolist()},
                colorscale: [[0, '#ff4757'], [0.5, '#ffa502'], [1, '#00ff88']]
            }}
        }};
        Plotly.newPlot('pnlChart', [pnlData], {{
            title: 'Top 20 Symbols by P&L',
            paper_bgcolor: '#16213e',
            plot_bgcolor: '#16213e',
            font: {{color: '#eee'}},
            xaxis: {{title: 'Symbol'}},
            yaxis: {{title: 'P&L ($)'}}
        }});
        
        // Equity Curve
        var equityData = {{
            x: {relaxed_df[relaxed_df['pnl'] != 0]['date'].tolist()},
            y: {relaxed_df[relaxed_df['pnl'] != 0]['pnl'].cumsum().tolist()},
            type: 'scatter',
            mode: 'lines',
            line: {{color: '#00d4ff', width: 2}}
        }};
        Plotly.newPlot('equityChart', [equityData], {{
            title: 'Cumulative P&L Over Time',
            paper_bgcolor: '#16213e',
            plot_bgcolor: '#16213e',
            font: {{color: '#eee'}},
            xaxis: {{title: 'Date'}},
            yaxis: {{title: 'Cumulative P&L ($)'}}
        }});
        
        // P&L Distribution
        var distData = {{
            x: {relaxed_df[relaxed_df['pnl'] != 0]['pnl'].tolist()},
            type: 'histogram',
            nbinsx: 30,
            marker: {{color: '#00d4ff'}}
        }};
        Plotly.newPlot('distributionChart', [distData], {{
            title: 'Trade P&L Distribution',
            paper_bgcolor: '#16213e',
            plot_bgcolor: '#16213e',
            font: {{color: '#eee'}},
            xaxis: {{title: 'P&L ($)'}},
            yaxis: {{title: 'Frequency'}}
        }});
    </script>
</body>
</html>
"""

# Save HTML report
report_path = output_dir / 'backtest_analysis.html'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"  [OK] Report saved to: {report_path}")

# Save JSON summary
with open(output_dir / 'analysis_summary.json', 'w') as f:
    json.dump(stats, f, indent=2)

# Print summary
print("\n[4/5] Generating summary tables...")
print("\n" + "="*80)
print("BACKTEST RESULTS SUMMARY")
print("="*80)
print(f"\n[DATA] Dataset: {stats['total_setups']:,} setups from {stats['unique_symbols']} symbols")
print(f"[DATE] Period: {stats['date_range']}")
print(f"\n[V5] V5 RELAXED PERFORMANCE:")
print(f"   Trades: {stats['v5']['trades_taken']}")
print(f"   Win Rate: {stats['v5']['win_rate']:.1f}%")
print(f"   Total P&L: ${stats['v5']['total_pnl']:,.0f}")
print(f"   Avg Trade: ${stats['v5']['avg_trade']:,.0f}")
print(f"   Max Win: ${stats['v5']['max_win']:,.0f}")
print(f"   Max Loss: ${stats['v5']['max_loss']:,.0f}")

if 'comparison' in stats:
    print(f"\n[ML] ML RISK COMPARISON:")
    print(f"   V5 Relaxed:  ${stats['comparison']['v5_pnl']:,.0f} ({stats['comparison']['v5_trades']} trades, {stats['comparison']['v5_win_rate']:.1f}% WR)")
    print(f"   V5 Inst:     ${stats['comparison']['inst_pnl']:,.0f} ({stats['comparison']['inst_trades']} trades, {stats['comparison']['inst_win_rate']:.1f}% WR)")
    diff = stats['comparison']['inst_pnl'] - stats['comparison']['v5_pnl']
    print(f"   Difference:  ${diff:+,.0f} ({'+' if diff > 0 else ''}{diff/stats['comparison']['v5_pnl']*100:.1f}%)")
    print(f"\n[WINNER] Winner: {'V5 Institutional' if diff > 0 else 'V5 Relaxed'}")

print("\n" + "="*80)
print("[DONE] ANALYSIS COMPLETE")
print(f"[REPORT] Report: {report_path.absolute()}")
print(f"[CHARTS] Charts saved to: {output_dir.absolute()}/")
print("="*80)
