"""
Historical Tick Data Fetcher for Alpaca
Retrieves actual trade and quote data for ultra-accurate backtesting.
"""
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

import polars as pl
import pandas as pd
import pytz

from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockTradesRequest, StockQuotesRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from src.utils.config import CONFIG
from src.utils.logger import logger


@dataclass
class HistoricalTick:
    """Single tick (trade or quote) with full market context."""
    timestamp: datetime
    symbol: str
    tick_type: str  # 'trade' or 'quote'
    
    # For trades
    trade_price: Optional[float] = None
    trade_size: Optional[int] = None
    trade_exchange: Optional[str] = None
    trade_conditions: Optional[List[str]] = None
    
    # For quotes
    bid_price: Optional[float] = None
    bid_size: Optional[int] = None
    ask_price: Optional[float] = None
    ask_size: Optional[int] = None
    quote_exchange: Optional[str] = None
    
    # Calculated fields
    mid_price: Optional[float] = None
    spread: Optional[float] = None


class HistoricalTickFetcher:
    """
    Fetches historical tick-level data (trades & quotes) from Alpaca.
    Provides much more accurate backtesting than bar data alone.
    """
    
    def __init__(self):
        self.client = StockHistoricalDataClient(
            api_key=CONFIG.broker.api_key,
            secret_key=CONFIG.broker.secret_key
        )
        self.cache_dir = Path("data/cache/ticks")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ny_tz = pytz.timezone('America/New_York')
    
    def _get_cache_path(self, symbol: str, date: datetime, data_type: str) -> Path:
        """Generate cache file path."""
        date_str = date.strftime("%Y%m%d")
        return self.cache_dir / f"{symbol}_{data_type}_{date_str}.parquet"
    
    def fetch_historical_trades(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        use_cache: bool = True
    ) -> pl.DataFrame:
        """
        Fetch historical trade data (actual executed transactions).
        
        Trade data includes:
        - timestamp: Nanosecond precision
        - price: Execution price
        - size: Number of shares
        - exchange: Where trade occurred
        - conditions: Trade conditions (e.g., '@', 'F', 'I')
        """
        # Ensure timezone-aware
        if start.tzinfo is None:
            start = self.ny_tz.localize(start)
        if end.tzinfo is None:
            end = self.ny_tz.localize(end)
        
        cache_path = self._get_cache_path(symbol, start, "trades")
        
        if use_cache and cache_path.exists():
            logger.info(f"Loading cached trades for {symbol}")
            return pl.read_parquet(cache_path)
        
        logger.info(f"Fetching historical trades for {symbol} ({start.date()})")
        
        try:
            request = StockTradesRequest(
                symbol_or_symbols=symbol,
                start=start,
                end=end,
                feed="iex"  # Free tier - use "sip" for pro
            )
            
            trades = self.client.get_stock_trades(request)
            
            # Convert to list of dicts
            data = []
            for trade in trades.data.get(symbol, []):
                data.append({
                    'timestamp': trade.timestamp,
                    'symbol': symbol,
                    'tick_type': 'trade',
                    'trade_price': trade.price,
                    'trade_size': trade.size,
                    'trade_exchange': trade.exchange,
                    'trade_conditions': ','.join(trade.conditions) if trade.conditions else '',
                    'bid_price': None,
                    'ask_price': None,
                    'mid_price': trade.price,
                    'spread': None
                })
            
            if not data:
                logger.warning(f"No trade data for {symbol}")
                return pl.DataFrame()
            
            df = pl.DataFrame(data)
            df = df.sort('timestamp')
            
            if use_cache:
                df.write_parquet(cache_path)
                logger.info(f"Cached {len(df):,} trades to {cache_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching trades for {symbol}: {e}")
            return pl.DataFrame()
    
    def fetch_historical_quotes(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        use_cache: bool = True
    ) -> pl.DataFrame:
        """
        Fetch historical quote data (bid/ask spreads).
        
        Quote data includes:
        - timestamp: Nanosecond precision
        - bid_price, bid_size: Best bid
        - ask_price, ask_size: Best ask
        - exchange: Quote exchange
        
        Note: Quote data is voluminous. For a liquid stock like AAPL,
        you might get millions of quotes per day.
        """
        if start.tzinfo is None:
            start = self.ny_tz.localize(start)
        if end.tzinfo is None:
            end = self.ny_tz.localize(end)
        
        cache_path = self._get_cache_path(symbol, start, "quotes")
        
        if use_cache and cache_path.exists():
            logger.info(f"Loading cached quotes for {symbol}")
            return pl.read_parquet(cache_path)
        
        logger.info(f"Fetching historical quotes for {symbol} ({start.date()})")
        
        try:
            request = StockQuotesRequest(
                symbol_or_symbols=symbol,
                start=start,
                end=end,
                feed="iex"
            )
            
            quotes = self.client.get_stock_quotes(request)
            
            data = []
            for quote in quotes.data.get(symbol, []):
                mid = (quote.bid_price + quote.ask_price) / 2 if quote.bid_price and quote.ask_price else None
                spread = quote.ask_price - quote.bid_price if quote.bid_price and quote.ask_price else None
                
                data.append({
                    'timestamp': quote.timestamp,
                    'symbol': symbol,
                    'tick_type': 'quote',
                    'trade_price': None,
                    'trade_size': None,
                    'trade_exchange': None,
                    'trade_conditions': None,
                    'bid_price': quote.bid_price,
                    'bid_size': quote.bid_size,
                    'ask_price': quote.ask_price,
                    'ask_size': quote.ask_size,
                    'quote_exchange': quote.exchange,
                    'mid_price': mid,
                    'spread': spread
                })
            
            if not data:
                return pl.DataFrame()
            
            df = pl.DataFrame(data)
            df = df.sort('timestamp')
            
            if use_cache:
                df.write_parquet(cache_path)
                logger.info(f"Cached {len(df):,} quotes to {cache_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching quotes for {symbol}: {e}")
            return pl.DataFrame()
    
    def fetch_combined_tick_data(
        self,
        symbol: str,
        date: datetime,
        use_quotes: bool = False,  # Quotes are huge, optional
        use_cache: bool = True
    ) -> pl.DataFrame:
        """
        Fetch combined trade and optionally quote data for a date.
        Returns unified DataFrame sorted by timestamp.
        """
        start = date.replace(hour=9, minute=30, second=0, microsecond=0)
        end = date.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Fetch trades (primary)
        trades_df = self.fetch_historical_trades(symbol, start, end, use_cache)
        
        if trades_df.is_empty():
            logger.warning(f"No trade data for {symbol} on {date.date()}")
            return pl.DataFrame()
        
        if use_quotes:
            # Fetch quotes and merge
            quotes_df = self.fetch_historical_quotes(symbol, start, end, use_cache)
            
            if not quotes_df.is_empty():
                # Combine and sort
                combined = pl.concat([trades_df, quotes_df])
                combined = combined.sort('timestamp')
                return combined
        
        return trades_df
    
    def aggregate_trades_to_bars(
        self,
        trades_df: pl.DataFrame,
        interval_seconds: int = 60
    ) -> pl.DataFrame:
        """
        Aggregate tick trade data into OHLCV bars.
        More accurate than downloaded bars because we see every trade.
        """
        if trades_df.is_empty():
            return pl.DataFrame()
        
        # Convert timestamps to ET timezone before aggregation
        # Alpaca returns UTC timestamps, we need ET for trading logic
        df = trades_df.with_columns([
            pl.col('timestamp').dt.convert_time_zone('America/New_York').alias('timestamp')
        ])
        
        # Create time buckets (in ET)
        df = df.with_columns([
            pl.col('timestamp').dt.truncate(f'{interval_seconds}s').alias('bar_time')
        ])
        
        # Aggregate to bars
        bars = df.group_by('bar_time').agg([
            pl.first('trade_price').alias('open'),
            pl.max('trade_price').alias('high'),
            pl.min('trade_price').alias('low'),
            pl.last('trade_price').alias('close'),
            pl.sum('trade_size').alias('volume'),
            pl.count().alias('trade_count'),
            # VWAP calculation - multiply price * size first, then aggregate
            (pl.col('trade_price') * pl.col('trade_size')).sum().alias('pv_sum'),
            pl.col('trade_size').sum().alias('size_sum')
        ])
        
        # Calculate VWAP from aggregated sums
        bars = bars.with_columns([
            (pl.col('pv_sum') / pl.col('size_sum')).alias('vwap')
        ])
        
        bars = bars.sort('bar_time')
        bars = bars.rename({'bar_time': 'timestamp'})
        
        return bars
    
    def detect_large_trades(
        self,
        trades_df: pl.DataFrame,
        min_size: int = 10000,
        min_dollar_volume: float = 500000
    ) -> pl.DataFrame:
        """
        Detect large institutional trades (potential iceberg orders).
        Returns trades that exceed the thresholds.
        """
        if trades_df.is_empty():
            return pl.DataFrame()
        
        large_trades = trades_df.with_columns([
            (pl.col('trade_price') * pl.col('trade_size')).alias('dollar_volume')
        ]).filter(
            (pl.col('trade_size') >= min_size) | 
            (pl.col('dollar_volume') >= min_dollar_volume)
        )
        
        return large_trades
    
    def calculate_market_impact(
        self,
        trades_df: pl.DataFrame,
        window_trades: int = 100
    ) -> pl.DataFrame:
        """
        Calculate market impact metrics:
        - Trade flow imbalance (buy vs sell pressure)
        - Volume at bid vs ask
        - Price impact per unit volume
        """
        if trades_df.is_empty() or len(trades_df) < window_trades:
            return pl.DataFrame()
        
        # Convert to pandas for rolling calculations
        pdf = trades_df.to_pandas()
        
        # Determine trade side (simplified - using tick rule)
        pdf['price_change'] = pdf['trade_price'].diff()
        pdf['side'] = pdf['price_change'].apply(
            lambda x: 'buy' if x > 0 else ('sell' if x < 0 else 'unknown')
        )
        
        # Calculate rolling metrics
        pdf['buy_volume'] = pdf[pdf['side'] == 'buy']['trade_size'].rolling(window=window_trades).sum()
        pdf['sell_volume'] = pdf[pdf['side'] == 'sell']['trade_size'].rolling(window=window_trades).sum()
        pdf['volume_imbalance'] = (pdf['buy_volume'] - pdf['sell_volume']) / (pdf['buy_volume'] + pdf['sell_volume'])
        
        return pl.from_pandas(pdf)
    
    def get_best_execution_price(
        self,
        trades_df: pl.DataFrame,
        side: str,  # 'buy' or 'sell'
        size: int,
        start_time: datetime,
        max_slippage_bps: float = 10.0  # Max 0.1% slippage
    ) -> Optional[float]:
        """
        Simulate order execution on historical tick data.
        Returns expected fill price considering market impact.
        """
        if trades_df.is_empty():
            return None
        
        # Filter trades after start_time
        mask = trades_df['timestamp'] >= start_time
        future_trades = trades_df.filter(mask)
        
        if future_trades.is_empty():
            return None
        
        # Get trades within slippage window
        reference_price = future_trades['trade_price'].first()
        max_price = reference_price * (1 + max_slippage_bps / 10000)
        min_price = reference_price * (1 - max_slippage_bps / 10000)
        
        valid_trades = future_trades.filter(
            (pl.col('trade_price') >= min_price) & 
            (pl.col('trade_price') <= max_price)
        )
        
        if valid_trades.is_empty():
            return reference_price  # Fallback
        
        # Calculate volume-weighted average price (VWAP) of available liquidity
        vwap = (valid_trades['trade_price'] * valid_trades['trade_size']).sum() / valid_trades['trade_size'].sum()
        
        return vwap


# Singleton
tick_fetcher = HistoricalTickFetcher()
