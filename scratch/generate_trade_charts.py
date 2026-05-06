"""
Generate Professional Trade Charts
Charts the top performing trades with entry/exit markers.
"""
import sys
from pathlib import Path
from datetime import datetime
import pickle

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.backtest.historical_tick_fetcher import tick_fetcher
from src.backtest.tick_backtest_engine_v5 import TickBacktestEngineV5
from src.backtest.backtest_engine import ActionType


# Top trades from our backtest (symbol, date, pnl)
TOP_TRADES = [
    ('GCT', '2022-08-19', 5676.87),
    ('GDC', '2024-08-21', 5366.11),
    ('RENT', '2024-04-11', 5113.24),
    ('CRBP', '2024-01-26', 4874.04),
    ('LIDR', '2024-05-10', 4713.72),
    ('WWR', '2020-10-05', 4116.77),
    ('HOUR', '2024-12-24', 4115.68),
    ('WKEY', '2024-12-13', 3489.05),
    ('OCGN', '2021-02-08', 3295.83),
    ('VANI', '2021-03-05', 3162.42),
]


def create_trade_chart(symbol: str, date_str: str, expected_pnl: float = None):
    """Create a professional TradingView-style chart for a trade."""
    
    date = datetime.strptime(date_str, "%Y-%m-%d")
    output_dir = Path("reports/charts")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generating chart for {symbol} on {date_str}")
    print(f"{'='*60}")
    
    # Fetch tick data
    tick_df = tick_fetcher.fetch_combined_tick_data(symbol, date, use_quotes=False)
    if tick_df.is_empty():
        print(f"  [NO DATA] No tick data available")
        return None
    
    # Aggregate to 1-minute bars
    bar_df = tick_fetcher.aggregate_trades_to_bars(tick_df, interval_seconds=60)
    if bar_df.is_empty():
        print(f"  [NO DATA] No bar data available")
        return None
    
    print(f"  [OK] Loaded {len(bar_df)} bars")
    
    # Run backtest to get entry/exit
    engine = TickBacktestEngineV5()
    result = engine.run_tick_backtest(symbol, date, verbose=False)
    
    # Extract trade details from audit records
    entries = []
    exits = []
    total_pnl = 0
    
    for record in result.audit_records:
        action_str = str(record.action) if hasattr(record, 'action') else ''
        
        if 'ENTRY' in action_str:
            entries.append({
                'time': record.timestamp,
                'price': record.price,
                'shares': record.shares
            })
            print(f"  [ENTRY] ${record.price:.2f} @ {record.timestamp.strftime('%H:%M')}")
            
        elif any(x in action_str for x in ['EXIT', 'TP1', 'TP2', 'TP3', 'STOP', 'TIME']):
            pnl = record.pnl if hasattr(record, 'pnl') else 0
            exit_price = record.exit_price if hasattr(record, 'exit_price') else record.price
            reason = record.exit_reason if hasattr(record, 'exit_reason') else 'exit'
            
            exits.append({
                'time': record.timestamp,
                'price': exit_price,
                'pnl': pnl,
                'reason': reason
            })
            total_pnl += pnl
            color = "[+]$" if pnl > 0 else "[-]$"
            print(f"  {color} Exit ({reason}): ${exit_price:.2f} | P&L: ${pnl:+.2f}")
    
    print(f"  [P&L] Total: ${total_pnl:+.2f}")
    
    # Prepare data for chart
    import pandas as pd
    bars = bar_df.to_pandas()
    bars['timestamp'] = pd.to_datetime(bars['timestamp'])
    bars = bars.sort_values('timestamp')
    
    # Calculate VWAP
    bars['typical'] = (bars['high'] + bars['low'] + bars['close']) / 3
    bars['tp_v'] = bars['typical'] * bars['volume']
    bars['cum_tp_v'] = bars['tp_v'].cumsum()
    bars['cum_vol'] = bars['volume'].cumsum()
    bars['vwap'] = bars['cum_tp_v'] / bars['cum_vol']
    
    # Calculate day statistics
    day_open = bars.iloc[0]['open']
    day_high = bars['high'].max()
    day_low = bars['low'].min()
    day_close = bars.iloc[-1]['close']
    day_gain = (day_high - day_open) / day_open * 100
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.65, 0.20, 0.15],
        subplot_titles=(
            f'<b>{symbol}</b> | {date_str} | Day Gain: {day_gain:.1f}%',
            'Volume',
            'Trade P&L'
        )
    )
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=bars['timestamp'],
        open=bars['open'],
        high=bars['high'],
        low=bars['low'],
        close=bars['close'],
        name='Price',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
        increasing_fillcolor='#26a69a',
        decreasing_fillcolor='#ef5350'
    ), row=1, col=1)
    
    # VWAP line
    fig.add_trace(go.Scatter(
        x=bars['timestamp'],
        y=bars['vwap'],
        name='VWAP',
        line=dict(color='#9c27b0', width=2),
        opacity=0.9
    ), row=1, col=1)
    
    # Entry markers
    for entry in entries:
        fig.add_trace(go.Scatter(
            x=[entry['time']],
            y=[entry['price']],
            mode='markers',
            marker=dict(
                symbol='triangle-up',
                size=20,
                color='#00c853',
                line=dict(width=2, color='white')
            ),
            name=f'Entry ${entry["price"]:.2f}',
            showlegend=True
        ), row=1, col=1)
    
    # Exit markers
    for exit in exits:
        color = '#ff1744' if exit['pnl'] < 0 else '#00c853'
        fig.add_trace(go.Scatter(
            x=[exit['time']],
            y=[exit['price']],
            mode='markers',
            marker=dict(
                symbol='triangle-down',
                size=20,
                color=color,
                line=dict(width=2, color='white')
            ),
            name=f'Exit ({exit["reason"]}) ${exit["pnl"]:+.0f}',
            showlegend=True
        ), row=1, col=1)
    
    # Volume bars
    colors = ['#26a69a' if c >= o else '#ef5350' 
              for c, o in zip(bars['close'], bars['open'])]
    fig.add_trace(go.Bar(
        x=bars['timestamp'],
        y=bars['volume'],
        name='Volume',
        marker_color=colors,
        opacity=0.7
    ), row=2, col=1)
    
    # P&L line
    if exits:
        cumulative = 0
        pnl_times = []
        pnl_values = []
        
        for exit in exits:
            cumulative += exit['pnl']
            pnl_times.append(exit['time'])
            pnl_values.append(cumulative)
        
        pnl_color = '#00c853' if cumulative > 0 else '#ff1744'
        fig.add_trace(go.Scatter(
            x=pnl_times,
            y=pnl_values,
            mode='lines+markers',
            name='Cumulative P&L',
            line=dict(color=pnl_color, width=2),
            fill='tozeroy',
            fillcolor=f'rgba(0, 200, 83, 0.1)' if cumulative > 0 else 'rgba(255, 23, 68, 0.1)'
        ), row=3, col=1)
    
    # Trade metrics box
    metrics_text = f"""
    <b>Trade Summary</b><br>
    P&L: <span style='color:{"#00c853" if total_pnl > 0 else "#ff1744"}'>${total_pnl:+,.2f}</span><br>
    Open: ${day_open:.2f}<br>
    High: ${day_high:.2f} ({day_gain:.1f}%)<br>
    Close: ${day_close:.2f}
    """
    
    fig.add_annotation(
        xref='paper', yref='paper',
        x=0.01, y=0.98,
        text=metrics_text,
        showarrow=False,
        font=dict(size=12, color='white'),
        bgcolor='rgba(30, 30, 30, 0.9)',
        bordercolor='gray',
        borderwidth=1,
        align='left',
        xanchor='left',
        yanchor='top'
    )
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f'<b>Parabolic Reversal Trade Analysis</b>',
            font=dict(size=20, color='white'),
            x=0.5
        ),
        paper_bgcolor='#131722',
        plot_bgcolor='#131722',
        font=dict(color='#d1d4dc', size=11),
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        height=950,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            bgcolor='rgba(0,0,0,0.5)',
            font=dict(size=10)
        ),
        margin=dict(t=100, b=50)
    )
    
    # Update axes
    for i in range(1, 4):
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', row=i, col=1)
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', row=i, col=1)
    
    fig.update_yaxes(title_text='Price ($)', row=1, col=1)
    fig.update_yaxes(title_text='Volume', row=2, col=1)
    fig.update_yaxes(title_text='P&L ($)', row=3, col=1)
    
    # Save
    filename = f"{symbol}_{date_str.replace('-', '')}_chart.html"
    filepath = output_dir / filename
    fig.write_html(str(filepath))
    print(f"  [SAVED] {filepath}\n")
    
    return filepath


def main():
    """Generate charts for top trades."""
    print("="*70)
    print("PROFESSIONAL TRADE CHART GENERATOR")
    print("TradingView-style visualization with entry/exit markers")
    print("="*70)
    
    generated = []
    for symbol, date, pnl in TOP_TRADES:
        try:
            filepath = create_trade_chart(symbol, date, pnl)
            if filepath:
                generated.append(filepath)
        except Exception as e:
            print(f"  [ERROR] {e}\n")
            continue
    
    print("\n" + "="*70)
    print(f"[SUCCESS] Generated {len(generated)} charts")
    print(f"Location: reports/charts/")
    print("="*70)
    print("\nOpen any HTML file in your browser to view the interactive charts.")


if __name__ == "__main__":
    main()
