"""
Generate Comprehensive Comparison Report

Creates detailed analysis comparing V5 Relaxed vs V5 Institutional ML strategies.

Output:
    - reports/comparison_report.html (interactive dashboard)
    - reports/comparison_charts/ (individual charts)
    - reports/comparison_analysis.xlsx (Excel analysis)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def load_data():
    """Load backtest results."""
    results_path = Path('reports/comparison_backtest_results.csv')
    summary_path = Path('reports/comparison_summary.json')
    
    if not results_path.exists():
        print("[ERROR] No results file found. Run backtest first.")
        return None, None
    
    df = pd.read_csv(results_path)
    
    summary = None
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary = json.load(f)
    
    return df, summary


def generate_equity_curves(df: pd.DataFrame):
    """Generate equity curve comparison."""
    print("[CHART] Generating equity curves...")
    
    # Calculate cumulative P&L for each strategy
    v5_df = df[df['strategy'] == 'v5_relaxed'].copy()
    inst_df = df[df['strategy'] == 'v5_institutional'].copy()
    
    v5_trades = v5_df[v5_df['pnl'] != 0].copy()
    inst_trades = inst_df[inst_df['pnl'] != 0].copy()
    
    if len(v5_trades) > 0:
        v5_trades['cumulative'] = v5_trades['pnl'].cumsum() + 100000
        v5_trades['trade_num'] = range(1, len(v5_trades) + 1)
    
    if len(inst_trades) > 0:
        inst_trades['cumulative'] = inst_trades['pnl'].cumsum() + 100000
        inst_trades['trade_num'] = range(1, len(inst_trades) + 1)
    
    # Create figure
    fig = go.Figure()
    
    if len(v5_trades) > 0:
        fig.add_trace(go.Scatter(
            x=v5_trades['trade_num'],
            y=v5_trades['cumulative'],
            mode='lines',
            name='V5 Relaxed',
            line=dict(color='#26a69a', width=2)
        ))
    
    if len(inst_trades) > 0:
        fig.add_trace(go.Scatter(
            x=inst_trades['trade_num'],
            y=inst_trades['cumulative'],
            mode='lines',
            name='V5 Institutional',
            line=dict(color='#1976d2', width=2)
        ))
    
    fig.add_hline(y=100000, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title='Equity Curve Comparison',
        xaxis_title='Trade Number',
        yaxis_title='Portfolio Value ($)',
        template='plotly_white',
        height=500
    )
    
    fig.write_html('reports/comparison_charts/equity_curves.html')
    print("  Saved: equity_curves.html")


def generate_monthly_performance(df: pd.DataFrame):
    """Generate monthly performance comparison."""
    print("[CHART] Generating monthly performance...")
    
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')
    
    monthly = df.groupby(['month', 'strategy'])['pnl'].sum().reset_index()
    monthly['month_str'] = monthly['month'].astype(str)
    
    fig = px.bar(
        monthly,
        x='month_str',
        y='pnl',
        color='strategy',
        barmode='group',
        title='Monthly P&L Comparison',
        labels={'pnl': 'P&L ($)', 'month_str': 'Month', 'strategy': 'Strategy'},
        color_discrete_map={
            'v5_relaxed': '#26a69a',
            'v5_institutional': '#1976d2'
        }
    )
    
    fig.update_layout(height=500, template='plotly_white')
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    fig.write_html('reports/comparison_charts/monthly_performance.html')
    print("  Saved: monthly_performance.html")


def generate_win_rate_comparison(df: pd.DataFrame):
    """Generate win rate comparison chart."""
    print("[CHART] Generating win rate comparison...")
    
    v5_df = df[df['strategy'] == 'v5_relaxed']
    inst_df = df[df['strategy'] == 'v5_institutional']
    
    v5_trades = v5_df[v5_df['pnl'] != 0]
    inst_trades = inst_df[inst_df['pnl'] != 0]
    
    v5_win_rate = v5_trades['win'].mean() * 100 if len(v5_trades) > 0 else 0
    inst_win_rate = inst_trades['win'].mean() * 100 if len(inst_trades) > 0 else 0
    
    fig = go.Figure(data=[
        go.Bar(
            name='Win Rate %',
            x=['V5 Relaxed', 'V5 Institutional'],
            y=[v5_win_rate, inst_win_rate],
            marker_color=['#26a69a', '#1976d2'],
            text=[f'{v5_win_rate:.1f}%', f'{inst_win_rate:.1f}%'],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Win Rate Comparison',
        yaxis_title='Win Rate (%)',
        template='plotly_white',
        height=400
    )
    
    fig.write_html('reports/comparison_charts/win_rate_comparison.html')
    print("  Saved: win_rate_comparison.html")


def generate_pnl_distribution(df: pd.DataFrame):
    """Generate P&L distribution comparison."""
    print("[CHART] Generating P&L distribution...")
    
    v5_df = df[df['strategy'] == 'v5_relaxed']
    inst_df = df[df['strategy'] == 'v5_institutional']
    
    v5_trades = v5_df[v5_df['pnl'] != 0]
    inst_trades = inst_df[inst_df['pnl'] != 0]
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('V5 Relaxed', 'V5 Institutional'))
    
    if len(v5_trades) > 0:
        fig.add_trace(
            go.Histogram(
                x=v5_trades['pnl'],
                nbinsx=30,
                name='V5 Relaxed',
                marker_color='#26a69a',
                opacity=0.7
            ),
            row=1, col=1
        )
    
    if len(inst_trades) > 0:
        fig.add_trace(
            go.Histogram(
                x=inst_trades['pnl'],
                nbinsx=30,
                name='V5 Institutional',
                marker_color='#1976d2',
                opacity=0.7
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        title='P&L Distribution Comparison',
        template='plotly_white',
        height=400
    )
    
    fig.write_html('reports/comparison_charts/pnl_distribution.html')
    print("  Saved: pnl_distribution.html")


def generate_drawdown_analysis(df: pd.DataFrame):
    """Generate drawdown analysis."""
    print("[CHART] Generating drawdown analysis...")
    
    # Calculate drawdowns
    def calc_drawdown(pnls):
        equity = np.cumsum(pnls) + 100000
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100
        return drawdown
    
    v5_df = df[df['strategy'] == 'v5_relaxed'].copy()
    inst_df = df[df['strategy'] == 'v5_institutional'].copy()
    
    v5_trades = v5_df[v5_df['pnl'] != 0]
    inst_trades = inst_df[inst_df['pnl'] != 0]
    
    fig = go.Figure()
    
    if len(v5_trades) > 0:
        v5_dd = calc_drawdown(v5_trades['pnl'].values)
        fig.add_trace(go.Scatter(
            y=v5_dd,
            mode='lines',
            name='V5 Relaxed',
            line=dict(color='#26a69a', width=1.5)
        ))
    
    if len(inst_trades) > 0:
        inst_dd = calc_drawdown(inst_trades['pnl'].values)
        fig.add_trace(go.Scatter(
            y=inst_dd,
            mode='lines',
            name='V5 Institutional',
            line=dict(color='#1976d2', width=1.5)
        ))
    
    fig.update_layout(
        title='Drawdown Analysis',
        yaxis_title='Drawdown (%)',
        xaxis_title='Trade Number',
        template='plotly_white',
        height=400
    )
    
    fig.write_html('reports/comparison_charts/drawdown_analysis.html')
    print("  Saved: drawdown_analysis.html")


def generate_summary_dashboard(df: pd.DataFrame, summary: dict):
    """Generate summary HTML dashboard."""
    print("[REPORT] Generating summary dashboard...")
    
    # Calculate key metrics
    v5_df = df[df['strategy'] == 'v5_relaxed']
    inst_df = df[df['strategy'] == 'v5_institutional']
    
    v5_trades = v5_df[v5_df['pnl'] != 0]
    inst_trades = inst_df[inst_df['pnl'] != 0]
    inst_blocked = len(inst_df[inst_df['pnl'] == 0])
    
    # Create HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Backtest Comparison Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background: #f5f5f5;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
            }}
            .header h1 {{
                margin: 0;
                font-size: 32px;
            }}
            .header p {{
                margin: 10px 0 0 0;
                opacity: 0.9;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .metric-card {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .metric-card h3 {{
                margin: 0 0 15px 0;
                color: #333;
                font-size: 18px;
            }}
            .metric-value {{
                font-size: 36px;
                font-weight: bold;
                color: #667eea;
            }}
            .metric-label {{
                color: #666;
                margin-top: 5px;
            }}
            .comparison-table {{
                width: 100%;
                background: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 30px;
            }}
            .comparison-table th {{
                background: #667eea;
                color: white;
                padding: 15px;
                text-align: left;
            }}
            .comparison-table td {{
                padding: 12px 15px;
                border-bottom: 1px solid #eee;
            }}
            .comparison-table tr:last-child td {{
                border-bottom: none;
            }}
            .positive {{
                color: #22c55e;
            }}
            .negative {{
                color: #ef4444;
            }}
            .charts-section {{
                background: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .charts-section h2 {{
                margin-top: 0;
                color: #333;
            }}
            .chart-links {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }}
            .chart-link {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                text-decoration: none;
                color: #667eea;
                transition: background 0.2s;
            }}
            .chart-link:hover {{
                background: #e9ecef;
            }}
            .recommendation {{
                background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                color: white;
                padding: 20px;
                border-radius: 8px;
                margin-top: 30px;
            }}
            .recommendation h2 {{
                margin: 0 0 10px 0;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Backtest Comparison Report</h1>
            <p>V5 Relaxed Scanner vs V5 Institutional ML Risk Management</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>V5 Relaxed</h3>
                <div class="metric-value">${v5_trades['pnl'].sum():,.0f}</div>
                <div class="metric-label">Total P&L ({len(v5_trades)} trades)</div>
            </div>
            <div class="metric-card">
                <h3>V5 Institutional</h3>
                <div class="metric-value">${inst_trades['pnl'].sum():,.0f}</div>
                <div class="metric-label">Total P&L ({len(inst_trades)} trades, {inst_blocked} blocked)</div>
            </div>
            <div class="metric-card">
                <h3>P&L Difference</h3>
                <div class="metric-value {'positive' if inst_trades['pnl'].sum() > v5_trades['pnl'].sum() else 'negative'}">
                    ${inst_trades['pnl'].sum() - v5_trades['pnl'].sum():+,.0f}
                </div>
                <div class="metric-label">Institutional vs Relaxed</div>
            </div>
        </div>
        
        <table class="comparison-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>V5 Relaxed</th>
                    <th>V5 Institutional</th>
                    <th>Difference</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Total Trades</td>
                    <td>{len(v5_trades)}</td>
                    <td>{len(inst_trades)}</td>
                    <td>{len(inst_trades) - len(v5_trades):+d}</td>
                </tr>
                <tr>
                    <td>Win Rate</td>
                    <td>{v5_trades['win'].mean()*100:.1f}%</td>
                    <td>{inst_trades['win'].mean()*100:.1f}%</td>
                    <td>{(inst_trades['win'].mean() - v5_trades['win'].mean())*100:+.1f}%</td>
                </tr>
                <tr>
                    <td>Average Trade</td>
                    <td>${v5_trades['pnl'].mean():,.0f}</td>
                    <td>${inst_trades['pnl'].mean():,.0f}</td>
                    <td>${inst_trades['pnl'].mean() - v5_trades['pnl'].mean():+,.0f}</td>
                </tr>
                <tr>
                    <td>Average Win</td>
                    <td>${v5_trades[v5_trades['pnl'] > 0]['pnl'].mean():,.0f}</td>
                    <td>${inst_trades[inst_trades['pnl'] > 0]['pnl'].mean():,.0f}</td>
                    <td>${inst_trades[inst_trades['pnl'] > 0]['pnl'].mean() - v5_trades[v5_trades['pnl'] > 0]['pnl'].mean():+,.0f}</td>
                </tr>
                <tr>
                    <td>Average Loss</td>
                    <td>${v5_trades[v5_trades['pnl'] < 0]['pnl'].mean():,.0f}</td>
                    <td>${inst_trades[inst_trades['pnl'] < 0]['pnl'].mean():,.0f}</td>
                    <td>${inst_trades[inst_trades['pnl'] < 0]['pnl'].mean() - v5_trades[v5_trades['pnl'] < 0]['pnl'].mean():+,.0f}</td>
                </tr>
                <tr>
                    <td>Max Win</td>
                    <td>${v5_trades['pnl'].max():,.0f}</td>
                    <td>${inst_trades['pnl'].max():,.0f}</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Max Loss</td>
                    <td>${v5_trades['pnl'].min():,.0f}</td>
                    <td>${inst_trades['pnl'].min():,.0f}</td>
                    <td>-</td>
                </tr>
            </tbody>
        </table>
        
        <div class="charts-section">
            <h2>Interactive Charts</h2>
            <div class="chart-links">
                <a href="comparison_charts/equity_curves.html" class="chart-link">Equity Curves</a>
                <a href="comparison_charts/monthly_performance.html" class="chart-link">Monthly Performance</a>
                <a href="comparison_charts/win_rate_comparison.html" class="chart-link">Win Rate Comparison</a>
                <a href="comparison_charts/pnl_distribution.html" class="chart-link">P&L Distribution</a>
                <a href="comparison_charts/drawdown_analysis.html" class="chart-link">Drawdown Analysis</a>
            </div>
        </div>
        
        <div class="recommendation">
            <h2>Recommendation</h2>
            <p>{summary.get('recommendation', 'Analyze results to determine best strategy') if summary else 'Run full backtest to generate recommendation'}</p>
        </div>
    </body>
    </html>
    """
    
    with open('reports/comparison_report.html', 'w') as f:
        f.write(html)
    
    print("  Saved: comparison_report.html")


def main():
    """Main report generation."""
    print("="*80)
    print("GENERATING COMPARISON REPORT")
    print("="*80)
    
    # Load data
    df, summary = load_data()
    
    if df is None:
        return
    
    print(f"\nLoaded {len(df)} result records")
    
    # Create charts directory
    Path('reports/comparison_charts').mkdir(exist_ok=True)
    
    # Generate charts
    generate_equity_curves(df)
    generate_monthly_performance(df)
    generate_win_rate_comparison(df)
    generate_pnl_distribution(df)
    generate_drawdown_analysis(df)
    
    # Generate dashboard
    generate_summary_dashboard(df, summary)
    
    print("\n" + "="*80)
    print("REPORT GENERATION COMPLETE")
    print("="*80)
    print("\nFiles generated:")
    print("  - reports/comparison_report.html (Main dashboard)")
    print("  - reports/comparison_charts/ (Individual charts)")
    print("\nOpen reports/comparison_report.html in your browser to view results.")


if __name__ == "__main__":
    main()
