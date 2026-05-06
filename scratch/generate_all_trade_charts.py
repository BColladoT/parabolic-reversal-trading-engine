"""
Generate charts for top 30 trades from relaxed backtest
"""
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.backtest.historical_tick_fetcher import tick_fetcher
from src.backtest.tick_backtest_engine_v5 import TickBacktestEngineV5
from src.backtest.backtest_engine import ActionType
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load results
df = pd.read_csv('reports/relaxed_909_backtest.csv')

# Get top 30 trades by P&L
top_trades = df.nlargest(30, 'pnl')

print("="*80)
print(f"GENERATING CHARTS FOR TOP 30 TRADES")
print("="*80)

output_dir = Path("reports/charts_relaxed")
output_dir.mkdir(parents=True, exist_ok=True)

generated = []

for idx, row in top_trades.iterrows():
    symbol = row['symbol']
    date_str = row['date']
    pnl = row['pnl']
    date = datetime.strptime(date_str, "%Y-%m-%d")
    
    print(f"\n[{len(generated)+1}/30] Charting {symbol} on {date_str} | P&L: ${pnl:+.2f}")
    
    try:
        # Fetch tick data
        tick_df = tick_fetcher.fetch_combined_tick_data(symbol, date, use_quotes=False)
        if tick_df.is_empty():
            print(f"  [SKIP] No tick data")
            continue
        
        # Aggregate to 1-minute bars
        bar_df = tick_fetcher.aggregate_trades_to_bars(tick_df, interval_seconds=60)
        if bar_df.is_empty():
            print(f"  [SKIP] No bar data")
            continue
        
        print(f"  [OK] Loaded {len(bar_df)} bars")
        
        # Run backtest to get entry/exit
        engine = TickBacktestEngineV5()
        result = engine.run_tick_backtest(symbol, date, verbose=False)
        
        # Extract trade details
        entries = []
        exits = []
        for record in result.audit_records:
            action_str = str(record.action) if hasattr(record, 'action') else ''
            
            if 'ENTRY' in action_str:
                entries.append({
                    'time': record.timestamp,
                    'price': record.price
                })
            elif any(x in action_str for x in ['EXIT', 'TP1', 'TP2', 'TP3', 'STOP', 'TIME']):
                pnl_trade = record.pnl if hasattr(record, 'pnl') else 0
                exit_price = record.exit_price if hasattr(record, 'exit_price') else record.price
                reason = record.exit_reason if hasattr(record, 'exit_reason') else 'exit'
                exits.append({
                    'time': record.timestamp,
                    'price': exit_price,
                    'pnl': pnl_trade,
                    'reason': reason
                })
        
        # Prepare data
        bars = bar_df.to_pandas()
        bars['timestamp'] = pd.to_datetime(bars['timestamp'])
        bars = bars.sort_values('timestamp')
        
        # Calculate VWAP
        bars['typical'] = (bars['high'] + bars['low'] + bars['close']) / 3
        bars['tp_v'] = bars['typical'] * bars['volume']
        bars['cum_tp_v'] = bars['tp_v'].cumsum()
        bars['cum_vol'] = bars['volume'].cumsum()
        bars['vwap'] = bars['cum_tp_v'] / bars['cum_vol']
        
        # Stats
        day_open = bars.iloc[0]['open']
        day_high = bars['high'].max()
        day_gain = (day_high - day_open) / day_open * 100
        
        # Create figure
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.65, 0.20, 0.15],
            subplot_titles=(
                f'{symbol} | {date_str} | Day Gain: {day_gain:.1f}% | Trade P&L: ${pnl:+.2f}',
                'Volume',
                'Trade P&L'
            )
        )
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=bars['timestamp'],
            open=bars['open'],
            high=bars['high'],
            low=bars['low'],
            close=bars['close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ), row=1, col=1)
        
        # VWAP
        fig.add_trace(go.Scatter(
            x=bars['timestamp'],
            y=bars['vwap'],
            name='VWAP',
            line=dict(color='#9c27b0', width=2)
        ), row=1, col=1)
        
        # Entry markers
        for entry in entries:
            fig.add_trace(go.Scatter(
                x=[entry['time']],
                y=[entry['price']],
                mode='markers',
                marker=dict(symbol='triangle-up', size=20, color='#00c853', line=dict(width=2, color='white')),
                name=f'Entry ${entry["price"]:.2f}'
            ), row=1, col=1)
        
        # Exit markers
        for exit in exits:
            color = '#ff1744' if exit['pnl'] < 0 else '#00c853'
            fig.add_trace(go.Scatter(
                x=[exit['time']],
                y=[exit['price']],
                mode='markers',
                marker=dict(symbol='triangle-down', size=20, color=color, line=dict(width=2, color='white')),
                name=f'Exit ({exit["reason"]}) ${exit["pnl"]:+.0f}'
            ), row=1, col=1)
        
        # Volume
        colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(bars['close'], bars['open'])]
        fig.add_trace(go.Bar(x=bars['timestamp'], y=bars['volume'], name='Volume', marker_color=colors, opacity=0.7), row=2, col=1)
        
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
                x=pnl_times, y=pnl_values, mode='lines+markers',
                name='Cumulative P&L', line=dict(color=pnl_color, width=2),
                fill='tozeroy', fillcolor=f'rgba(0,200,83,0.1)' if cumulative > 0 else 'rgba(255,23,68,0.1)'
            ), row=3, col=1)
        
        # Layout
        fig.update_layout(
            title=dict(text=f'Parabolic Reversal Trade Analysis | Rank #{len(generated)+1}', font=dict(size=20, color='white'), x=0.5),
            paper_bgcolor='#131722',
            plot_bgcolor='#131722',
            font=dict(color='#d1d4dc', size=11),
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            height=950,
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, bgcolor='rgba(0,0,0,0.5)', font=dict(size=10)),
            margin=dict(t=100, b=50)
        )
        
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
        print(f"  [SAVED] {filepath}")
        generated.append(filepath)
        
    except Exception as e:
        print(f"  [ERROR] {e}")
        continue

print("\n" + "="*80)
print(f"Generated {len(generated)} charts")
print(f"Location: {output_dir}/")
print("="*80)
