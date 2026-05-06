"""
Backtest Visualization Module
Creates charts and HTML reports for backtest results.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from src.backtest.backtest_engine import BacktestResult, AuditRecord, ActionType


class BacktestVisualizer:
    """Creates visualizations for backtest results."""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def create_equity_curve(self, result: BacktestResult, save_path: str = None):
        """Plot equity curve with trade markers."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Build equity curve
        equity = []
        timestamps = []
        current_equity = 100000  # Starting capital
        
        for record in result.audit_records:
            if record.action == ActionType.ENTRY:
                current_equity -= record.risk_amount
            elif record.action == ActionType.EXIT and record.pnl is not None:
                current_equity += record.risk_amount + record.pnl
            
            equity.append(current_equity)
            timestamps.append(record.timestamp)
        
        # Plot equity curve
        ax1.plot(timestamps, equity, 'b-', linewidth=1.5, label='Equity')
        ax1.axhline(y=100000, color='gray', linestyle='--', alpha=0.5, label='Start')
        
        # Mark trades
        for record in result.audit_records:
            if record.action == ActionType.ENTRY:
                ax1.scatter(record.timestamp, current_equity, 
                          color='red', marker='v', s=100, zorder=5, label='Entry')
            elif record.action == ActionType.EXIT and record.pnl is not None:
                color = 'green' if record.pnl > 0 else 'red'
                ax1.scatter(record.timestamp, current_equity,
                          color=color, marker='^', s=100, zorder=5)
        
        ax1.set_title(f'Equity Curve - {result.symbol}', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Equity ($)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot P&L distribution
        pnls = [r.pnl for r in result.audit_records 
                if r.action == ActionType.EXIT and r.pnl is not None]
        
        if pnls:
            colors = ['green' if p > 0 else 'red' for p in pnls]
            ax2.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.set_title('Trade P&L Distribution', fontsize=12)
            ax2.set_xlabel('Trade Number', fontsize=10)
            ax2.set_ylabel('P&L ($)', fontsize=10)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved chart to {save_path}")
        
        return fig
    
    def create_trade_chart(self, result: BacktestResult, price_data: pd.DataFrame, 
                          save_path: str = None):
        """
        Create detailed price chart with entry/exit markers and indicators.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), 
                                       gridspec_kw={'height_ratios': [3, 1]},
                                       sharex=True)
        
        # Price and VWAP
        ax1.plot(price_data.index, price_data['close'], 
                'b-', linewidth=1, label='Price', alpha=0.8)
        
        if 'vwap' in price_data.columns:
            ax1.plot(price_data.index, price_data['vwap'],
                    'orange', linewidth=1.5, label='VWAP', linestyle='--')
        
        # Mark entries and exits
        for record in result.audit_records:
            if record.action == ActionType.ENTRY:
                ax1.scatter(record.timestamp, record.price,
                          color='red', marker='v', s=200, zorder=5,
                          label='Short Entry', edgecolors='black', linewidths=1)
                
                # Add stop loss and target lines
                ax1.axhline(y=record.stop_loss, color='red', linestyle=':', 
                          alpha=0.5, xmin=0, xmax=1)
                ax1.axhline(y=record.profit_target, color='green', linestyle=':',
                          alpha=0.5, xmin=0, xmax=1)
                
            elif record.action == ActionType.EXIT:
                color = 'green' if record.pnl and record.pnl > 0 else 'red'
                label = 'Cover (Win)' if record.pnl and record.pnl > 0 else 'Cover (Loss)'
                ax1.scatter(record.timestamp, record.exit_price,
                          color=color, marker='^', s=200, zorder=5,
                          label=label, edgecolors='black', linewidths=1)
        
        ax1.set_title(f'{result.symbol} - Price Action with Trades', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Volume
        if 'volume' in price_data.columns:
            colors = ['green' if price_data['close'].iloc[i] >= price_data['open'].iloc[i] 
                     else 'red' for i in range(len(price_data))]
            ax2.bar(price_data.index, price_data['volume'], color=colors, alpha=0.6, width=0.001)
            ax2.set_ylabel('Volume', fontsize=12)
            ax2.set_xlabel('Time', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved chart to {save_path}")
        
        return fig
    
    def generate_html_report(self, result: BacktestResult, output_path: str = None) -> str:
        """Generate interactive HTML report."""
        if output_path is None:
            output_path = self.output_dir / f"backtest_{result.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        # Build trade table rows
        trade_rows = []
        for i, record in enumerate(result.audit_records):
            if record.action == ActionType.ENTRY:
                trade_rows.append(f"""
                <tr class="entry-row">
                    <td>{record.timestamp.strftime('%H:%M:%S')}</td>
                    <td><span class="badge entry">ENTRY</span></td>
                    <td>${record.price:.2f}</td>
                    <td>-</td>
                    <td>{record.position_size}</td>
                    <td>{record.confirming_factors}/4</td>
                    <td>{record.confidence_score:.0%}</td>
                    <td colspan="2" class="reasoning">{record.reasoning}</td>
                </tr>
                """)
            elif record.action == ActionType.EXIT:
                pnl_class = "positive" if record.pnl and record.pnl > 0 else "negative"
                pnl_sign = "+" if record.pnl and record.pnl > 0 else ""
                trade_rows.append(f"""
                <tr class="exit-row">
                    <td>{record.timestamp.strftime('%H:%M:%S')}</td>
                    <td><span class="badge exit">EXIT</span></td>
                    <td>-</td>
                    <td>${record.exit_price:.2f}</td>
                    <td>-</td>
                    <td>-</td>
                    <td>-</td>
                    <td class="pnl {pnl_class}">{pnl_sign}${record.pnl:.2f}</td>
                    <td>{record.exit_reason}</td>
                </tr>
                """)
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report - {result.symbol}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
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
            font-size: 28px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
            color: #333;
        }}
        .metric-label {{
            color: #666;
            font-size: 14px;
            margin-top: 5px;
        }}
        .positive {{ color: #22c55e; }}
        .negative {{ color: #ef4444; }}
        .trades-section {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th {{
            background: #f8f9fa;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            color: #666;
            border-bottom: 2px solid #dee2e6;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #dee2e6;
        }}
        .entry-row {{ background: #fef2f2; }}
        .exit-row {{ background: #f0fdf4; }}
        .badge {{
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }}
        .badge.entry {{ background: #fee2e2; color: #dc2626; }}
        .badge.exit {{ background: #dcfce7; color: #16a34a; }}
        .reasoning {{
            font-size: 12px;
            color: #666;
            max-width: 400px;
        }}
        .audit-log {{
            margin-top: 30px;
            background: white;
            padding: 20px;
            border-radius: 8px;
        }}
        .audit-entry {{
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #667eea;
            background: #f8f9fa;
        }}
        .audit-entry h4 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .audit-details {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            font-size: 13px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Backtest Report: {result.symbol}</h1>
        <p>Period: {result.start_date.date()} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-value {('positive' if result.total_pnl > 0 else 'negative')}">${result.total_pnl:,.2f}</div>
            <div class="metric-label">Total P&L</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{result.total_trades}</div>
            <div class="metric-label">Total Trades</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{result.win_rate:.1%}</div>
            <div class="metric-label">Win Rate</div>
        </div>
        <div class="metric-card">
            <div class="metric-value {('positive' if result.profit_factor > 1 else 'negative')}">{result.profit_factor:.2f}</div>
            <div class="metric-label">Profit Factor</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">${result.average_trade:,.2f}</div>
            <div class="metric-label">Average Trade</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">${result.average_win:,.2f} / ${result.average_loss:,.2f}</div>
            <div class="metric-label">Avg Win / Loss</div>
        </div>
    </div>
    
    <div class="trades-section">
        <h2>Trade Log with Reasoning</h2>
        <table>
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Action</th>
                    <th>Entry Price</th>
                    <th>Exit Price</th>
                    <th>Size</th>
                    <th>Factors</th>
                    <th>Confidence</th>
                    <th>P&L</th>
                    <th>Exit Reason</th>
                </tr>
            </thead>
            <tbody>
                {''.join(trade_rows)}
            </tbody>
        </table>
    </div>
    
    <div class="audit-log">
        <h2>Detailed Audit Log</h2>
        {self._generate_audit_html(result.audit_records)}
    </div>
</body>
</html>
"""
        
        with open(output_path, 'w') as f:
            f.write(html)
        
        print(f"HTML report saved to {output_path}")
        return str(output_path)
    
    def _generate_audit_html(self, records: List[AuditRecord]) -> str:
        """Generate HTML for detailed audit entries."""
        entries = []
        for record in records:
            if record.action == ActionType.ENTRY:
                entries.append(f"""
                <div class="audit-entry">
                    <h4>[{record.timestamp.strftime('%H:%M:%S')}] ENTRY {record.symbol} @ ${record.price:.2f}</h4>
                    <div class="audit-details">
                        <div><strong>VWAP Extension:</strong> {record.vwap_extension:.2f}x</div>
                        <div><strong>ATR:</strong> ${record.atr:.2f}</div>
                        <div><strong>Volume:</strong> {record.volume:,} ({record.volume_vs_avg:.1f}x avg)</div>
                        <div><strong>Volume Exhaustion:</strong> {'Yes' if record.volume_exhaustion else 'No'}</div>
                        <div><strong>Momentum Divergence:</strong> {'Yes' if record.momentum_divergence else 'No'}</div>
                        <div><strong>Absorption:</strong> {'Yes' if record.absorption_detected else 'No'}</div>
                    </div>
                    <p><strong>Reasoning:</strong> {record.reasoning}</p>
                    <p><strong>Position:</strong> {record.position_size} shares | 
                       <strong>Risk:</strong> ${record.risk_amount:.2f} | 
                       <strong>Stop:</strong> ${record.stop_loss:.2f} | 
                       <strong>Target:</strong> ${record.profit_target:.2f}</p>
                </div>
                """)
        return '\n'.join(entries)


# Singleton
visualizer = BacktestVisualizer()
