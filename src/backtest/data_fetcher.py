"""
Historical Data Fetcher for Backtesting
Downloads and caches historical data from Alpaca and other sources.
"""
import os
import json
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

import polars as pl
import numpy as np
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockQuotesRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import yfinance as yf

from src.utils.config import CONFIG
from src.utils.logger import logger


@dataclass
class HistoricalBar:
    """Historical bar data for backtesting."""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    trades: int = 0


class DataFetcher:
    """
    Fetches and caches historical market data for backtesting.
    Supports Alpaca (primary) and Yahoo Finance (fallback).
    """
    
    def __init__(self):
        self.client = StockHistoricalDataClient(
            api_key=CONFIG.broker.api_key,
            secret_key=CONFIG.broker.secret_key
        )
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_path(self, symbol: str, start: datetime, end: datetime, 
                        timeframe: str) -> Path:
        """Generate cache file path."""
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")
        return self.cache_dir / f"{symbol}_{timeframe}_{start_str}_{end_str}.parquet"
    
    def fetch_alpaca_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: TimeFrame = TimeFrame.Minute,
        use_cache: bool = True
    ) -> pl.DataFrame:
        """
        Fetch historical bars from Alpaca API.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        start, end : datetime
            Date range
        timeframe : TimeFrame
            Bar granularity (default: 1 minute)
        use_cache : bool
            Use cached data if available
            
        Returns:
        --------
        pl.DataFrame with OHLCV data
        """
        cache_path = self._get_cache_path(symbol, start, end, "1min")
        
        # Check cache
        if use_cache and cache_path.exists():
            logger.info(f"Loading cached data for {symbol}")
            return pl.read_parquet(cache_path)
        
        logger.info(f"Fetching {symbol} data from Alpaca ({start.date()} to {end.date()})")
        
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                feed="iex"  # Free tier
            )
            
            bars = self.client.get_stock_bars(request)
            
            # Convert to Polars DataFrame
            data = []
            for bar in bars.data.get(symbol, []):
                data.append({
                    'timestamp': bar.timestamp,
                    'symbol': symbol,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume,
                    'vwap': bar.vwap if hasattr(bar, 'vwap') else None,
                    'trades': bar.trade_count if hasattr(bar, 'trade_count') else 0
                })
            
            if not data:
                logger.warning(f"No data returned for {symbol}")
                return pl.DataFrame()
            
            df = pl.DataFrame(data)
            
            # Sort by timestamp
            df = df.sort('timestamp')
            
            # Cache to parquet
            if use_cache:
                df.write_parquet(cache_path)
                logger.info(f"Cached {len(df)} bars to {cache_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return pl.DataFrame()
    
    def fetch_yahoo(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1m"
    ) -> pl.DataFrame:
        """
        Fetch data from Yahoo Finance (fallback for free historical data).
        Note: Yahoo has 7-day limit for 1m data, 60 days for hourly.
        """
        logger.info(f"Fetching {symbol} from Yahoo Finance")
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, interval=interval)
            
            if df.empty:
                return pl.DataFrame()
            
            # Reset index to make datetime a column
            df = df.reset_index()
            
            # Rename columns
            df = df.rename(columns={
                'Datetime': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            df['symbol'] = symbol
            df['vwap'] = None
            df['trades'] = 0
            
            return pl.from_pandas(df)
            
        except Exception as e:
            logger.error(f"Yahoo fetch error: {e}")
            return pl.DataFrame()
    
    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: TimeFrame = TimeFrame.Minute
    ) -> Dict[str, pl.DataFrame]:
        """Fetch data for multiple symbols."""
        results = {}
        for symbol in symbols:
            df = self.fetch_alpaca_bars(symbol, start, end, timeframe)
            if not df.is_empty():
                results[symbol] = df
        return results
    
    def find_parabolic_candidates(
        self,
        symbols: List[str],
        lookback_days: int = 30,
        min_gain_percent: float = 80.0
    ) -> List[Dict]:
        """
        Scan historical data to find parabolic moves for backtesting.
        Returns list of candidates with their parabolic day data.
        """
        end = datetime.now()
        start = end - timedelta(days=lookback_days)
        
        candidates = []
        
        for symbol in symbols:
            df = self.fetch_alpaca_bars(symbol, start, end, TimeFrame.Day)
            
            if df.is_empty() or len(df) < 2:
                continue
            
            # Calculate daily gains
            df = df.with_columns([
                (((pl.col('close') / pl.col('open')) - 1) * 100).alias('gain_percent')
            ])
            
            # Find days with parabolic gains
            parabolic_days = df.filter(pl.col('gain_percent') >= min_gain_percent)
            
            if not parabolic_days.is_empty():
                for row in parabolic_days.to_dicts():
                    candidates.append({
                        'symbol': symbol,
                        'date': row['timestamp'],
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume'],
                        'gain_percent': row['gain_percent']
                    })
        
        # Sort by gain percent
        candidates.sort(key=lambda x: x['gain_percent'], reverse=True)
        return candidates
    
    def get_intraday_for_date(
        self,
        symbol: str,
        date: datetime,
        use_cache: bool = True
    ) -> pl.DataFrame:
        """
        Get minute-level data for a specific date (for backtesting a trade).
        """
        start = date.replace(hour=0, minute=0, second=0)
        end = date.replace(hour=23, minute=59, second=59)
        
        return self.fetch_alpaca_bars(symbol, start, end, TimeFrame.Minute, use_cache)


# Singleton instance
data_fetcher = DataFetcher()
