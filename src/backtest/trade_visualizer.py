"""
Professional Trade Visualization - TradingView Style Charts
Shows entry/exit points, P&L, and key metrics on interactive charts.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import polars as pl
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pickle

from src.backtest.historical_tick_fetcher import tick_fetcher
from src.backtest.tick_backtest_engine_v5 import TickBacktestEngineV5
from src.backtest.historical_screener import ParabolicSetup


class TradeVisualizer:
    """
    Creates professional TradingView-style charts for backtest trades.
    """
    
    def __init__(self, output_dir: str = "reports/charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def chart_trade(self, symbol: str, date: datetime, setup: Dict = None, 
                    save_html: bool = True, show_browser: bool = False) -> go.Figure:
        """
        Create a professional chart for a single trade with:
        - Candlestick price action
        - VWAP overlay
        - Entry/exit markers
        - Volume bars
        - Trade metrics annotation
        """
        # Fetch tick data
        tick_df = tick_fetcher.fetch_combined_tick_data(symbol, date, use_quotes=False)
        if tick_df.is_empty():
            print(f"No data for {symbol} on {date.date()}")
            return None
            
        # Aggregate to 1-minute bars
        bar_df = tick_fetcher.aggregate_trades_to_bars(tick_df, interval_seconds=60)
        if bar_df.is_empty():
            return None
            
        # Convert to pandas for easier plotting
        bars = bar_df.to_pandas()
        bars['timestamp'] = pd.to_datetime(bars['timestamp'])
        bars = bars.sort_values('timestamp')
        
        # Calculate VWAP
        bars['typical_price'] = (bars['high'] + bars['low'] + bars['close']) / 3
        bars['tp_v'] = bars['typical_price'] * bars['volume']
        bars['cumulative_tp_v'] = bars['tp_v'].cumsum()
        bars['cumulative_vol'] = bars['volume'].cumsum()
        bars['vwap'] = bars['cumulative_tp_v'] / bars['cumulative_vol']
        
        # Run backtest to get entry/exit points
        engine = TickBacktestEngineV5()
        result = engine.run_tick_backtest(symbol, date, verbose=False)
        
        # Extract entry/exit points from audit records
        entries = []
        exits = []
        for record in result.audit_records:
            if hasattr(record, 'action'):
                if 'ENTRY' in str(record.action):
                    entries.append({
                        'time': record.timestamp,
                        'price': record.price,
                        'shares': record.shares
                    })
                elif 'EXIT' in str(record.action) or record.action.__class__.__name__ in ['TP1_EXIT', 'TP2_EXIT', 'TP3_EXIT', 'STOP_EXIT', 'TIME_EXIT']:
                    exits.append({
                        'time': record.timestamp,
                        'price': record.exit_price if hasattr(record, 'exit_price') else record.price,
                        'pnl': record.pnl if hasattr(record, 'pnl') else 0,
                        'reason': record.exit_reason if hasattr(record, 'exit_reason') else 'exit'
                    })
        
        # Create figure with subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(f'{symbol} - {date.strftime("%Y-%m-%d")}', 'Volume', 'Trade P&L')
        )
        
        # Main candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=bars['timestamp'],
                open=bars['open'],
                high=bars['high'],
                low=bars['low'],
                close=bars['close'],
                name='Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )
        
        # VWAP line
        fig.add_trace(
            go.Scatter(
                x=bars['timestamp'],
                y=bars['vwap'],
                name='VWAP',
                line=dict(color='#9c27b0', width=2, dash='dot'),
                opacity=0.8
            ),
            row=1, col=1
        )
        
        # Entry markers (green triangles up)
        for entry in entries:
            fig.add_trace(
                go.Scatter(
                    x=[entry['time']],
                    y=[entry['price']],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=15,
                        color='#00c853',
                        line=dict(width=2, color='white')
                    ),
                    name=f'Entry @ ${entry["price"]:.2f}',
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Exit markers (red triangles down)
        for exit in exits:
            color = '#ff1744' if exit['pnl'] < 0 else '#00c853'
            fig.add_trace(
                go.Scatter(
                    x=[exit['time']],
                    y=[exit['price']],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=15,
                        color=color,
                        line=dict(width=2, color='white')
                    ),
                    name=f'Exit ({exit["reason"]}) @ ${exit["price"]:.2f} | P&L: ${exit["pnl"]:+.2f}',
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Volume bars
        colors = ['#26a69a' if bars.iloc[i]['close'] >= bars.iloc[i]['open'] else '#ef5350' 
                  for i in range(len(bars))]
        
        fig.add_trace(
            go.Bar(
                x=bars['timestamp'],
                y=bars['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # P&L line (if we have trade data)
        if exits:
            cumulative_pnl = 0
            pnl_times = []
            pnl_values = []
            
            for exit in exits:
                cumulative_pnl += exit['pnl']
                pnl_times.append(exit['time'])
                pnl_values.append(cumulative_pnl)
            
            fig.add_trace(
                go.Scatter(
                    x=pnl_times,
                    y=pnl_values,
                    mode='lines+markers',
                    name='Cumulative P&L',
                    line=dict(color='#2196f3', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(33, 150, 243, 0.2)'
                ),
                row=3, col=1
            )
        
        # Add trade metrics annotation
        if result.total_trades > 0:
            win_rate = result.winning_trades / result.total_trades * 100
            metrics_text = f"""
            <b>Trade Metrics</b><br>
            Total P&L: ${result.total_pnl:+.2f}<br>
            Win Rate: {win_rate:.0f}%<br>
            Trades: {result.total_trades}<br>
            Avg Trade: ${result.average_trade:+.2f}
            """
            
            fig.add_annotation(
                xref='paper', yref='paper',
                x=0.02, y=0.98,
                text=metrics_text,
                showarrow=False,
                font=dict(size=12, color='white'),
                bgcolor='rgba(0,0,0,0.7)',
                bordercolor='gray',
                borderwidth=1,
                align='left',
                xanchor='left',
                yanchor='top'
            )
        
        # Layout configuration
        fig.update_layout(
            title=dict(
                text=f'<b>{symbol}</b> Parabolic Reversal Trade Analysis | {date.strftime("%Y-%m-%d")}',
                font=dict(size=18),
                x=0.5
            ),
            paper_bgcolor='#131722',
            plot_bgcolor='#131722',
            font=dict(color='#d1d4dc'),
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            height=900,
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                bgcolor='rgba(0,0,0,0.5)'
            )
        )
        
        # Update y-axes colors
        fig.update_yaxes(title_text='Price ($)', gridcolor='rgba(255,255,255,0.1)', row=1, col=1)
        fig.update_yaxes(title_text='Volume', gridcolor='rgba(255,255,255,0.1)', row=2, col=1)
        fig.update_yaxes(title_text='P&L ($)', gridcolor='rgba(255,255,255,0.1)', row=3, col=1)
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', rangeslider=dict(visible=False))
        
        # Save to HTML
        if save_html:
            filename = f"{symbol}_{date.strftime('%Y%m%d')}_trade.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))
            print(f"Chart saved: {filepath}")
        
        return fig
    
    def chart_all_trades(self, trades_csv: str = "reports/batch_backtest_20260311_124411.csv"):
        """
        Generate charts for all trades in the CSV report.
        """
        if not Path(trades_csv).exists():
            print(f"Trade file not found: {trades_csv}")
            # Try to find latest
            report_dir = Path("reports")
            csv_files = list(report_dir.glob("batch_backtest_*.csv"))
            if csv_files:
                trades_csv = str(sorted(csv_files)[-1])
                print(f"Using latest: {trades_csv}")
            else:
                return
        
        df = pd.read_csv(trades_csv)
        print(f"Generating charts for {len(df)} trades...")
        
        for idx, row in df.iterrows():
            try:
                symbol = row['symbol']
                date = pd.to_datetime(row['date'])
                
                print(f"Charting {symbol} on {date.date()}...")
                self.chart_trade(symbol, date)
                
            except Exception as e:
                print(f"Error charting {row.get('symbol', 'unknown')}: {e}")
                continue
        
        print(f"\nAll charts saved to: {self.output_dir}/")


# Convenience function
def chart_top_trades(n: int = 10):
    """
    Chart the top N performing trades.
    """
    viz = TradeVisualizer()
    
    # Use all_setups_backtest.csv which has detailed data
    report_file = Path("reports/all_setups_backtest.csv")
    if not report_file.exists():
        print(f"Report not found: {report_file}")
        return
    
    df = pd.read_csv(report_file)
    
    # Filter to trades with actual P&L data (non-empty P&L column)
    df['P&L_num'] = pd.to_numeric(df['P&L'], errors='coerce')
    df_trades = df[df['P&L_num'].notna() & (df['P&L_num'] != 0)].copy()
    
    if len(df_trades) == 0:
        print("No trades with P&L found in report")
        return
    
    # Sort by P&L and take top N
    df_sorted = df_trades.sort_values('P&L_num', ascending=False).head(n)
    
    print(f"Charting top {n} trades from {len(df_trades)} total trades...")
    print(f"Data source: {report_file}\n")
    
    for idx, row in df_sorted.iterrows():
        try:
            symbol = row['Symbol']
            date = pd.to_datetime(row['Date'])
            pnl = row['P&L_num']
            
            print(f"  {symbol} ({date.date()}): ${pnl:+.2f}")
            viz.chart_trade(symbol, date)
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    print(f"\nCharts saved to: {viz.output_dir}/")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--top":
            n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            chart_top_trades(n)
        elif sys.argv[1] == "--all":
            viz = TradeVisualizer()
            viz.chart_all_trades()
        elif sys.argv[1] == "--symbol":
            # Chart specific symbol/date
            symbol = sys.argv[2]
            date_str = sys.argv[3] if len(sys.argv) > 3 else "2024-08-21"
            date = datetime.strptime(date_str, "%Y-%m-%d")
            viz = TradeVisualizer()
            viz.chart_trade(symbol, date)
    else:
        # Default: chart top 10 winners
        chart_top_trades(10)
