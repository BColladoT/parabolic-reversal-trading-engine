"""
Polars-Based High-Performance Data Engine
Zero-copy data processing with Apache Arrow backend.
Optimized for sub-millisecond signal generation.
"""
import polars as pl
import pyarrow as pa
import numpy as np
from typing import Optional, Dict, List, Deque
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import threading

from src.utils.config import CONFIG
from src.utils.logger import logger
from src.indicators.numba_kernels import (
    calculate_vwap_numba,
    calculate_atr_numba,
    calculate_atr_latest_numba,
    calculate_vwap_incremental_numba
)

# Polars configuration for maximum performance
pl.Config.set_tbl_rows(20)
if CONFIG.performance.polars_threads > 0:
    pl.Config.set_threadpool_size(CONFIG.performance.polars_threads)


@dataclass
class TickData:
    """Raw tick data structure."""
    timestamp: datetime
    symbol: str
    price: float
    size: int
    side: str  # 'B' or 'A' (Bid/Ask)
    exchange: str


@dataclass  
class BarData:
    """Aggregated bar data."""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float
    trades: int


class StreamingBuffer:
    """
    Ring buffer for tick data with automatic aggregation.
    Uses PyArrow for zero-copy transfers to Polars.
    """
    
    def __init__(self, symbol: str, max_size: int = 10000):
        self.symbol = symbol
        self.max_size = max_size
        self.ticks: Deque[TickData] = deque(maxlen=max_size)
        self._lock = threading.RLock()
        
        # VWAP state
        self.cum_pv = 0.0
        self.cum_vol = 0.0
        self.current_vwap = 0.0
        
        # Bar aggregation state
        self.current_bar: Optional[BarData] = None
        self.bar_history: List[BarData] = []
        self.last_bar_time: Optional[datetime] = None
        
    def add_tick(self, tick: TickData) -> Optional[BarData]:
        """
        Add tick to buffer and return completed bar if new bar formed.
        """
        with self._lock:
            self.ticks.append(tick)
            
            # Update incremental VWAP
            self.current_vwap, self.cum_pv, self.cum_vol = calculate_vwap_incremental_numba(
                self.current_vwap,
                self.cum_pv,
                self.cum_vol,
                tick.price,
                tick.price,
                tick.price,
                float(tick.size)
            )
            
            # Aggregate into bars
            return self._aggregate_bar(tick)
    
    def _aggregate_bar(self, tick: TickData) -> Optional[BarData]:
        """Aggregate ticks into time-based bars."""
        bar_interval = CONFIG.data.bar_aggregation_seconds
        
        # Floor timestamp to bar interval
        tick_seconds = tick.timestamp.second + tick.timestamp.microsecond / 1e6
        bar_start_second = int(tick_seconds / bar_interval) * bar_interval
        bar_time = tick.timestamp.replace(second=int(bar_start_second), microsecond=0)
        
        # Check if we need to close previous bar
        if self.last_bar_time is not None and bar_time > self.last_bar_time:
            completed_bar = self.current_bar
            if completed_bar:
                self.bar_history.append(completed_bar)
                # Trim history to prevent memory bloat
                if len(self.bar_history) > 500:
                    self.bar_history = self.bar_history[-200:]
            self.current_bar = None
        
        # Update or create current bar
        if self.current_bar is None or bar_time > self.last_bar_time:
            self.current_bar = BarData(
                timestamp=bar_time,
                symbol=self.symbol,
                open=tick.price,
                high=tick.price,
                low=tick.price,
                close=tick.price,
                volume=tick.size,
                vwap=self.current_vwap,
                trades=1
            )
            self.last_bar_time = bar_time
        else:
            # Update existing bar
            bar = self.current_bar
            bar.high = max(bar.high, tick.price)
            bar.low = min(bar.low, tick.price)
            bar.close = tick.price
            bar.volume += tick.size
            bar.vwap = self.current_vwap
            bar.trades += 1
        
        return None
    
    def to_polars_df(self, n_ticks: Optional[int] = None) -> pl.DataFrame:
        """
        Convert tick buffer to Polars DataFrame with zero-copy.
        """
        with self._lock:
            if not self.ticks:
                return pl.DataFrame()
            
            ticks_to_use = list(self.ticks)[-n_ticks:] if n_ticks else list(self.ticks)
            
            # Build Arrow arrays directly
            timestamps = pa.array([t.timestamp for t in ticks_to_use])
            prices = pa.array([t.price for t in ticks_to_use])
            sizes = pa.array([t.size for t in ticks_to_use])
            sides = pa.array([t.side for t in ticks_to_use])
            
            arrow_table = pa.table({
                'timestamp': timestamps,
                'price': prices,
                'size': sizes,
                'side': sides
            })
            
            # Zero-copy conversion to Polars
            return pl.from_arrow(arrow_table)
    
    def get_bar_df(self, n_bars: int = 100) -> pl.DataFrame:
        """Get bar history as Polars DataFrame."""
        with self._lock:
            bars = self.bar_history[-n_bars:]
            if not bars:
                return pl.DataFrame()
            
            return pl.DataFrame({
                'timestamp': [b.timestamp for b in bars],
                'open': [b.open for b in bars],
                'high': [b.high for b in bars],
                'low': [b.low for b in bars],
                'close': [b.close for b in bars],
                'volume': [b.volume for b in bars],
                'vwap': [b.vwap for b in bars],
                'trades': [b.trades for b in bars]
            })
    
    def get_latest_metrics(self) -> Dict:
        """Get latest metrics for signal generation."""
        with self._lock:
            if not self.ticks:
                return {}
            
            latest_tick = self.ticks[-1]
            recent_ticks = list(self.ticks)[-100:]
            
            prices = np.array([t.price for t in recent_ticks])
            volumes = np.array([float(t.size) for t in recent_ticks])
            
            metrics = {
                'symbol': self.symbol,
                'last_price': latest_tick.price,
                'last_size': latest_tick.size,
                'vwap': self.current_vwap,
                'vwap_extension': latest_tick.price / self.current_vwap if self.current_vwap > 0 else 1.0,
                'tick_count': len(self.ticks),
                'high_24h': np.max(prices) if len(prices) > 0 else latest_tick.price,
                'low_24h': np.min(prices) if len(prices) > 0 else latest_tick.price,
                'volume_sum': np.sum(volumes) if len(volumes) > 0 else 0,
                'price_volatility': np.std(prices) if len(prices) > 1 else 0.0
            }
            
            # Add ATR if we have bar history
            if len(self.bar_history) >= 14:
                bar_df = self.get_bar_df(n_bars=50)
                if len(bar_df) >= 14:
                    atr = calculate_atr_latest_numba(
                        bar_df['high'].to_numpy(),
                        bar_df['low'].to_numpy(),
                        bar_df['close'].to_numpy(),
                        period=14
                    )
                    metrics['atr'] = atr
                    metrics['atr_percent'] = (atr / latest_tick.price) * 100 if latest_tick.price > 0 else 0
            
            return metrics


class PolarsSignalEngine:
    """
    High-performance signal generation using Polars LazyFrames.
    """
    
    def __init__(self):
        self.buffers: Dict[str, StreamingBuffer] = {}
        
    def register_symbol(self, symbol: str):
        """Register a symbol for streaming."""
        if symbol not in self.buffers:
            self.buffers[symbol] = StreamingBuffer(
                symbol=symbol,
                max_size=CONFIG.data.tick_buffer_size
            )
            logger.info(f"Registered symbol buffer: {symbol}")
    
    def process_tick(self, tick: TickData) -> Optional[BarData]:
        """Process incoming tick data."""
        if tick.symbol not in self.buffers:
            self.register_symbol(tick.symbol)
        
        return self.buffers[tick.symbol].add_tick(tick)
    
    def get_signal_data(self, symbol: str) -> Dict:
        """Get all data needed for signal generation."""
        if symbol not in self.buffers:
            return {}
        
        buffer = self.buffers[symbol]
        metrics = buffer.get_latest_metrics()
        
        # Build LazyFrame for complex queries
        bar_df = buffer.get_bar_df(n_bars=100)
        
        if len(bar_df) > 0:
            # Use LazyFrame for efficient computation
            lazy_df = bar_df.lazy()
            
            # Calculate rolling metrics efficiently
            metrics['price_change_5bar'] = (
                lazy_df.select(
                    ((pl.col('close') / pl.col('close').shift(5)) - 1) * 100
                ).collect().to_numpy()[-1][0]
                if len(bar_df) >= 5 else 0
            )
            
            metrics['volume_trend'] = (
                lazy_df.select(
                    pl.col('volume').tail(5).mean() / pl.col('volume').head(5).mean()
                ).collect().to_numpy()[-1][0]
                if len(bar_df) >= 10 else 1.0
            )
            
            # Detect volume exhaustion
            recent_vol = bar_df['volume'].tail(3).mean()
            peak_vol = bar_df['volume'].max()
            metrics['volume_exhaustion'] = recent_vol < (peak_vol * CONFIG.signals.volume_exhaustion_factor)
        
        return metrics
    
    def calculate_vwap_full(self, symbol: str) -> Optional[pl.DataFrame]:
        """Calculate full VWAP history for a symbol."""
        if symbol not in self.buffers:
            return None
        
        bar_df = self.buffers[symbol].get_bar_df(n_bars=500)
        if len(bar_df) < 2:
            return None
        
        # Convert to numpy for Numba processing
        highs = bar_df['high'].to_numpy()
        lows = bar_df['low'].to_numpy()
        closes = bar_df['close'].to_numpy()
        volumes = bar_df['volume'].to_numpy()
        
        # Calculate VWAP using Numba kernel
        vwap_values = calculate_vwap_numba(highs, lows, closes, volumes)
        
        # Return as Polars DataFrame
        return bar_df.with_columns([
            pl.Series('vwap_calculated', vwap_values)
        ])
    
    def cleanup_old_data(self, max_age_hours: int = 8):
        """Clean up old data to prevent memory leaks."""
        # Reset VWAP state for new sessions
        for buffer in self.buffers.values():
            buffer.bar_history = buffer.bar_history[-200:]  # Keep last 200 bars
