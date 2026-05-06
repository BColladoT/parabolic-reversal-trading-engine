"""
Data Provider for RL Training - ALL PARQUET FILES VERSION

Loads ALL available 1-minute OHLCV intraday data from Parquet files.
Instead of filtering for specific setups, uses all trading days from all symbols.
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, time, date as dt_date
import random
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import pickle

logger = logging.getLogger(__name__)


@dataclass 
class MarketBar:
    """Single bar of market data with engineered features."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float
    vwap_deviation: float
    volume_concentration: float
    
    def __repr__(self):
        return (f"MarketBar({self.timestamp}, O:{self.open:.2f}, H:{self.high:.2f}, "
                f"L:{self.low:.2f}, C:{self.close:.2f}, VWAP:{self.vwap:.2f}, "
                f"Dev:{self.vwap_deviation:.1f}%, VolConc:{self.volume_concentration:.2f})")


class TradingDayData:
    """All bars for a single trading day with metadata."""
    
    def __init__(self, symbol: str, date: datetime, bars: List[MarketBar]):
        self.symbol = symbol
        self.date = date
        self.bars = bars
        
    def __len__(self) -> int:
        return len(self.bars)
    
    def get_bar_at_idx(self, idx: int) -> Optional[MarketBar]:
        """Get bar at specific index."""
        if 0 <= idx < len(self.bars):
            return self.bars[idx]
        return None


class HistoricalDataProvider:
    """
    Data provider that loads ALL intraday 1-min OHLCV from Parquet files.
    
    Scans all available Parquet files and extracts every trading day for training.
    This gives the model exposure to all market conditions, not just "successful" setups.
    """
    
    def __init__(
        self,
        intraday_data_dir: str = "data/cache/1min_extended",
        cache_index_path: str = "data/cache/trading_days_index.pkl",
        symbol_filter: Optional[List[str]] = None,
        date_range: Optional[Tuple[str, str]] = None,  # ("2020-01-01", "2024-12-31")
        min_bars_per_day: int = 100,  # Minimum bars to consider a valid trading day
        scan_parallel: bool = True
    ):
        """
        Initialize data provider.
        
        Args:
            intraday_data_dir: Directory containing *_1min_*.parquet files
            cache_index_path: Path to cache the (symbol, date) index
            symbol_filter: Optional list of symbols to include (None = all)
            date_range: Optional (start_date, end_date) to filter
            min_bars_per_day: Minimum number of bars required for a valid day
            scan_parallel: Whether to scan files in parallel
        """
        self.intraday_data_dir = Path(intraday_data_dir)
        self.cache_index_path = Path(cache_index_path)
        self.symbol_filter = set(symbol_filter) if symbol_filter else None
        self.date_range = date_range
        self.min_bars_per_day = min_bars_per_day
        
        self.current_day: Optional[TradingDayData] = None
        self.current_bar_idx: int = 0
        self.episode_count: int = 0
        
        # Cache of available symbols and their files
        self._symbol_file_map: Dict[str, Path] = {}
        
        # List of (symbol, date) tuples from ALL parquet files
        self.setup_pairs: List[Tuple[str, str]] = []
        
        # Statistics
        self.stats = {
            'files_scanned': 0,
            'symbols_found': 0,
            'trading_days_found': 0,
            'dates_loaded': set()
        }
        
        self._scan_intraday_files()
        
        # Try to load cached index first, otherwise scan and cache
        if not self._load_cached_index():
            self._build_trading_days_index(scan_parallel)
            self._save_cached_index()
        
        logger.info(f"Data provider ready: {len(self.setup_pairs)} total trading days available")
        
    def _scan_intraday_files(self):
        """Scan intraday data directory to build symbol->file mapping."""
        if not self.intraday_data_dir.exists():
            logger.warning(f"Intraday data directory not found: {self.intraday_data_dir}")
            return
            
        # Look for files like: SYMBOL_1min_YYYYMMDD_YYYYMMDD.parquet
        for file_path in self.intraday_data_dir.glob("*_1min_*.parquet"):
            # Extract symbol from filename (e.g., "AAPL_1min_20190101_20241231.parquet")
            symbol = file_path.name.split('_')[0]
            
            # Apply symbol filter if specified
            if self.symbol_filter and symbol not in self.symbol_filter:
                continue
                
            self._symbol_file_map[symbol] = file_path
            
        self.stats['files_scanned'] = len(self._symbol_file_map)
        logger.info(f"Scanned {len(self._symbol_file_map)} symbols in {self.intraday_data_dir}")
        
    def _load_cached_index(self) -> bool:
        """Try to load cached (symbol, date) index."""
        if not self.cache_index_path.exists():
            return False
            
        try:
            with open(self.cache_index_path, 'rb') as f:
                cache = pickle.load(f)
                
            # Verify cache is for the same directory
            if cache.get('data_dir') != str(self.intraday_data_dir):
                logger.info("Cache index is for different directory, rebuilding...")
                return False
                
            self.setup_pairs = cache.get('setup_pairs', [])
            self.stats = cache.get('stats', self.stats)
            
            logger.info(f"Loaded cached index: {len(self.setup_pairs)} trading days")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load cached index: {e}")
            return False
            
    def _save_cached_index(self):
        """Save (symbol, date) index to cache file."""
        try:
            self.cache_index_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_index_path, 'wb') as f:
                pickle.dump({
                    'data_dir': str(self.intraday_data_dir),
                    'setup_pairs': self.setup_pairs,
                    'stats': self.stats,
                    'created_at': datetime.now().isoformat()
                }, f)
            logger.info(f"Cached index saved to {self.cache_index_path}")
        except Exception as e:
            logger.warning(f"Failed to save cached index: {e}")
            
    def _extract_dates_from_parquet(self, symbol: str, file_path: Path) -> List[str]:
        """
        Extract trading dates with HIGH VOLATILITY from a parquet file.
        
        Only includes days where:
        1. Minimum number of bars (liquid trading day)
        2. Price range > 20% (high volatility - parabolic candidate)
        
        Returns:
            List of date strings in format "YYYY-MM-DD"
        """
        try:
            # Read necessary columns for volatility calculation
            df = pl.read_parquet(file_path, columns=['timestamp', 'open', 'high', 'low', 'close'])
            
            if len(df) == 0:
                return []
                
            # Handle different timestamp formats
            timestamp_col = 'timestamp'
            ts_dtype = df[timestamp_col].dtype
            
            if ts_dtype == pl.Utf8:
                df = df.with_columns([
                    pl.col(timestamp_col).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
                    .alias('_ts')
                ])
            elif ts_dtype == pl.Date:
                df = df.with_columns([
                    pl.col(timestamp_col).cast(pl.Datetime).alias('_ts')
                ])
            elif 'Datetime' in str(ts_dtype) or 'timestamp' in str(ts_dtype).lower():
                df = df.with_columns([
                    pl.col(timestamp_col).dt.replace_time_zone(None).alias('_ts')
                ])
            else:
                df = df.with_columns([
                    pl.col(timestamp_col).cast(pl.Datetime).alias('_ts')
                ])
            
            # Extract date component
            df = df.with_columns([
                pl.col('_ts').dt.date().alias('_date')
            ])
            
            # Calculate metrics per day
            daily_metrics = df.group_by('_date').agg([
                pl.count().alias('bar_count'),
                pl.min('low').alias('day_low'),
                pl.max('high').alias('day_high'),
                pl.first('open').alias('day_open'),
                pl.last('close').alias('day_close')
            ])
            
            # Filter for high-volatility days
            valid_dates = []
            for row in daily_metrics.iter_rows(named=True):
                date_obj = row['_date']
                
                # Check minimum bars (liquid trading day)
                if row['bar_count'] < self.min_bars_per_day:
                    continue
                
                # Check date range if specified
                if self.date_range:
                    start_date = datetime.strptime(self.date_range[0], "%Y-%m-%d").date()
                    end_date = datetime.strptime(self.date_range[1], "%Y-%m-%d").date()
                    if not (start_date <= date_obj <= end_date):
                        continue
                
                # Calculate daily gain from open
                if row['day_open'] > 0:
                    day_gain_pct = ((row['day_close'] - row['day_open']) / row['day_open']) * 100
                    day_range_pct = ((row['day_high'] - row['day_low']) / row['day_open']) * 100
                    
                    # ONLY include high volatility days (> 20% gain OR > 30% intraday range)
                    # This ensures the agent sees parabolic moves worth trading
                    if day_gain_pct > 20 or day_range_pct > 30:
                        valid_dates.append(date_obj.isoformat())
                
            return valid_dates
            
        except Exception as e:
            logger.warning(f"Error extracting dates from {file_path}: {e}")
            return []
            
    def _build_trading_days_index(self, parallel: bool = True):
        """Build index of all (symbol, date) pairs from all parquet files."""
        logger.info(f"Building trading days index from {len(self._symbol_file_map)} files...")
        
        all_pairs = []
        
        if parallel and len(self._symbol_file_map) > 10:
            # Parallel scanning for many files
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {
                    executor.submit(self._extract_dates_from_parquet, symbol, path): symbol 
                    for symbol, path in self._symbol_file_map.items()
                }
                
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        dates = future.result()
                        for date_str in dates:
                            all_pairs.append((symbol, date_str))
                        if dates:
                            logger.debug(f"{symbol}: {len(dates)} trading days")
                    except Exception as e:
                        logger.warning(f"Error processing {symbol}: {e}")
        else:
            # Sequential scanning
            for symbol, file_path in sorted(self._symbol_file_map.items()):
                dates = self._extract_dates_from_parquet(symbol, file_path)
                for date_str in dates:
                    all_pairs.append((symbol, date_str))
                if len(dates) > 0:
                    logger.debug(f"{symbol}: {len(dates)} trading days")
                    
        self.setup_pairs = all_pairs
        self.stats['trading_days_found'] = len(all_pairs)
        self.stats['symbols_found'] = len(set(p[0] for p in all_pairs))
        self.stats['dates_loaded'] = set(p[1] for p in all_pairs)
        
        logger.info(f"Index complete: {self.stats['trading_days_found']} trading days "
                   f"from {self.stats['symbols_found']} symbols")
        
        if len(all_pairs) == 0:
            logger.error("No trading days found! Check your data directory and date range.")
            
    def _engineer_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Engineer features from raw OHLCV data with computational safety nets.
        
        Calculates:
        1. Intraday Anchored VWAP: Cumulative VWAP from market open (9:30 AM ET)
        2. VWAP Deviation: Percentage deviation from VWAP
        3. Volume Concentration: Current volume vs 20-period rolling average
        
        SAFETY NETS:
        - Epsilon (1e-8) prevents division by zero (Inf)
        - fill_null() prevents NaN from EMA warmup
        - clip() prevents extreme outliers from destabilizing SAC
        
        TIMEZONE HANDLING:
        - Raw Alpaca data is timezone-naive, treated as UTC
        - First localize to UTC, then convert to America/New_York
        
        Args:
            df: Polars DataFrame with OHLCV columns (must have '_ts' datetime column)
            
        Returns:
            DataFrame with engineered features added (GUARANTEED no NaN/Inf)
        """
        # Safety epsilon for division operations
        EPS = 1e-8
        
        # CRITICAL: Alpaca raw data is timezone-naive
        # Must first localize to UTC, then convert to ET
        # Chain: naive → UTC → America/New_York
        df = df.with_columns([
            pl.col('_ts')
            .dt.replace_time_zone('UTC')           # Step 1: Treat naive as UTC
            .dt.convert_time_zone('America/New_York')  # Step 2: Convert to ET
            .dt.hour()
            .cast(pl.Int32)
            .alias('et_hour'),
            
            pl.col('_ts')
            .dt.replace_time_zone('UTC')           # Step 1: Treat naive as UTC
            .dt.convert_time_zone('America/New_York')  # Step 2: Convert to ET
            .dt.minute()
            .cast(pl.Int32)
            .alias('et_minute')
        ])
        
        # Identify bars after market open (9:30 AM = 570 minutes from midnight)
        df = df.with_columns([
            ((pl.col('et_hour') * 60 + pl.col('et_minute')) >= 570).alias('after_open')
        ])
        
        # Calculate typical price = (high + low + close) / 3
        df = df.with_columns([
            ((pl.col('high') + pl.col('low') + pl.col('close')) / 3).alias('typical_price')
        ])
        
        # Calculate price * volume for VWAP numerator
        df = df.with_columns([
            (pl.col('typical_price') * pl.col('volume')).alias('typical_pv')
        ])
        
        # Calculate cumulative VWAP from market open using numpy for conditional cumsum
        after_open = df['after_open'].to_numpy()
        typical_pv = df['typical_pv'].to_numpy()
        volume = df['volume'].to_numpy()
        close = df['close'].to_numpy()
        
        cum_pv = 0.0
        cum_vol = 0.0
        vwap_values = []
        
        for i in range(len(df)):
            if after_open[i]:
                cum_pv += typical_pv[i]
                cum_vol += volume[i]
                # Safety: if cum_vol is 0 (shouldn't happen but protect anyway), use close
                vwap_values.append(cum_pv / (cum_vol + EPS) if cum_vol > 0 else close[i])
            else:
                # Before market open, use close price
                vwap_values.append(close[i])
        
        df = df.with_columns([
            pl.Series('vwap', vwap_values)
        ])
        
        # Calculate VWAP deviation: ((close - vwap) / (vwap + EPS)) * 100
        # EPS prevents division by zero when vwap is 0 (rare but possible)
        df = df.with_columns([
            ((pl.col('close') - pl.col('vwap')) / (pl.col('vwap') + EPS) * 100)
            .alias('vwap_deviation')
        ])
        
        # Clip VWAP deviation to prevent extreme outliers (e.g., -500%, +1000%)
        # Micro-caps can have wild moves, but SAC needs bounded inputs
        df = df.with_columns([
            pl.col('vwap_deviation').clip(-200.0, 200.0).alias('vwap_deviation')
        ])
        
        # Calculate Volume Concentration: current volume / 20-period EMA
        # Use exponential moving average for smoother results
        df = df.with_columns([
            pl.col('volume').ewm_mean(span=20).alias('volume_ema_20')
        ])
        
        # Volume concentration with epsilon to prevent division by zero
        df = df.with_columns([
            (pl.col('volume') / (pl.col('volume_ema_20') + EPS)).alias('volume_concentration')
        ])
        
        # SAFETY NET 1: Fill NaN values from EMA warmup (first 19 bars)
        # CRITICAL: No backward fill to prevent look-ahead bias
        # Use forward fill only (propagate first valid value forward)
        # Then fill any remaining with neutral baseline (1.0 = normal volume)
        df = df.with_columns([
            pl.col('volume_concentration').fill_null(strategy='forward').fill_null(1.0)
        ])
        
        # If still null (entire column is null), fill with 1.0 (neutral)
        df = df.with_columns([
            pl.col('volume_concentration').fill_null(1.0)
        ])
        
        # SAFETY NET 2: Clip extreme outliers
        # Micro-cap capitulation can cause concentration of 50x-100x normal volume
        # SAC needs bounded inputs; clip to reasonable range [0, 10]
        df = df.with_columns([
            pl.col('volume_concentration').clip(0.0, 10.0).alias('volume_concentration')
        ])
        
        # SAFETY NET 3: Final NaN/Inf check using numpy
        # Convert to numpy, replace any remaining invalid values, convert back
        vwap_dev_np = df['vwap_deviation'].to_numpy()
        vol_conc_np = df['volume_concentration'].to_numpy()
        
        # Replace NaN with 0 (VWAP deviation) or 1 (volume concentration)
        vwap_dev_np = np.nan_to_num(vwap_dev_np, nan=0.0, posinf=200.0, neginf=-200.0)
        vol_conc_np = np.nan_to_num(vol_conc_np, nan=1.0, posinf=10.0, neginf=0.0)
        
        df = df.with_columns([
            pl.Series('vwap_deviation', vwap_dev_np),
            pl.Series('volume_concentration', vol_conc_np)
        ])
        
        # Clean up intermediate columns
        df = df.drop(['et_hour', 'et_minute', 'after_open', 'typical_price', 
                      'typical_pv', 'volume_ema_20'])
        
        # FINAL GUARANTEE: Assert no NaN or Inf values exist
        assert not df['vwap_deviation'].is_null().any(), "NaN detected in vwap_deviation!"
        assert not df['volume_concentration'].is_null().any(), "NaN detected in volume_concentration!"
        assert not df['vwap'].is_null().any(), "NaN detected in vwap!"
        
        return df
    
    def _load_trading_day(self, symbol: str, date_str: str) -> Optional[TradingDayData]:
        """Load full intraday 1-minute OHLCV data for a trading day with feature engineering."""
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return None
            
        # Find the parquet file for this symbol
        file_path = self._symbol_file_map.get(symbol)
        if not file_path:
            logger.debug(f"No cache file for symbol: {symbol}")
            return None
            
        try:
            # Read the parquet file
            df = pl.read_parquet(file_path)
            
            if len(df) == 0:
                logger.debug(f"Empty file for {symbol}")
                return None
            
            # Find timestamp column - try common names
            timestamp_col = None
            for col in df.columns:
                col_lower = col.lower()
                if col_lower in ['timestamp', 'datetime', 'date', 'time']:
                    timestamp_col = col
                    break
            
            if timestamp_col is None:
                # Try any column that might be a timestamp
                for col in df.columns:
                    if any(keyword in col.lower() for keyword in ['time', 'date']):
                        timestamp_col = col
                        break
            
            if timestamp_col is None:
                logger.warning(f"No timestamp column found in {file_path.name}")
                return None
            
            # DEBUG: Log what we found
            logger.debug(f"Found timestamp column: {timestamp_col}, dtype: {df[timestamp_col].dtype}")
            
            # Convert timestamp to datetime - handle various formats
            ts_dtype = df[timestamp_col].dtype
            
            if ts_dtype == pl.Utf8:
                # Try parsing as datetime string
                df = df.with_columns([
                    pl.col(timestamp_col).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
                    .alias('_ts')
                ])
            elif ts_dtype == pl.Date:
                df = df.with_columns([
                    pl.col(timestamp_col).cast(pl.Datetime).alias('_ts')
                ])
            elif 'Datetime' in str(ts_dtype) or 'timestamp' in str(ts_dtype).lower():
                # Already datetime type (might have timezone)
                df = df.with_columns([
                    pl.col(timestamp_col).dt.replace_time_zone(None).alias('_ts')
                ])
            else:
                # Try casting
                df = df.with_columns([
                    pl.col(timestamp_col).cast(pl.Datetime).alias('_ts')
                ])
            
            # Extract date component for filtering
            df = df.with_columns([
                pl.col('_ts').dt.date().alias('_date')
            ])
            
            # Filter to target date
            target_date = date.date()
            df_filtered = df.filter(pl.col('_date') == target_date)
            
            if len(df_filtered) == 0:
                logger.warning(f"No data for {symbol} on {date_str}. Available dates: {df['_date'].min()} to {df['_date'].max()}")
                return None
            
            logger.info(f"Found {len(df_filtered)} raw rows for {symbol} on {date_str}")
            
            # Sort by timestamp
            df_filtered = df_filtered.sort('_ts')
            
            # Find OHLCV columns - be flexible with naming
            col_map = {}
            for req in ['open', 'high', 'low', 'close', 'volume']:
                for col in df.columns:
                    if req == col.lower() or req in col.lower():
                        col_map[req] = col
                        break
                        
            if len(col_map) < 5:
                logger.warning(f"Missing OHLCV columns for {symbol} {date_str}: found {col_map}")
                return None
            
            # Rename columns to standard names for feature engineering
            df_filtered = df_filtered.rename({
                col_map['open']: 'open',
                col_map['high']: 'high',
                col_map['low']: 'low',
                col_map['close']: 'close',
                col_map['volume']: 'volume'
            })
            
            # Apply feature engineering (VWAP, VWAP deviation, volume concentration)
            df_filtered = self._engineer_features(df_filtered)
            
            # Convert to MarketBar objects
            bars = []
            for row in df_filtered.iter_rows(named=True):
                ts = row['_ts']
                # Handle different timestamp types
                if hasattr(ts, 'to_pydatetime'):
                    ts = ts.to_pydatetime()
                elif isinstance(ts, str):
                    ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    
                bar = MarketBar(
                    timestamp=ts,
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=float(row['volume']),
                    vwap=float(row['vwap']),
                    vwap_deviation=float(row['vwap_deviation']),
                    volume_concentration=float(row['volume_concentration'])
                )
                bars.append(bar)
            
            logger.info(f"Successfully loaded {len(bars)} engineered bars for {symbol} {date_str}")
            return TradingDayData(symbol, date, bars)
            
        except Exception as e:
            logger.error(f"Error loading {symbol} {date_str}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
            
    def reset_episode(self, specific_setup: Optional[Dict] = None) -> bool:
        """Reset for a new episode - load a trading day."""
        self.episode_count += 1
        
        if specific_setup:
            symbol = specific_setup['symbol']
            date = specific_setup['date']
        else:
            if not self.setup_pairs:
                logger.warning("No setup pairs available")
                return False
            symbol, date = random.choice(self.setup_pairs)
            
        # Load the trading day data
        day_data = self._load_trading_day(symbol, date)
        
        if day_data is None:
            logger.warning(f"Failed to load intraday data for {symbol} on {date}")
            return False
            
        self.current_day = day_data
        self.current_bar_idx = 0
        
        logger.info(f"Episode {self.episode_count}: {symbol} {date} ({len(day_data)} bars)")
        return True
        
    def get_current_bar(self) -> Optional[MarketBar]:
        """Get current bar without advancing."""
        if self.current_day is None:
            return None
        return self.current_day.get_bar_at_idx(self.current_bar_idx)
        
    def advance(self) -> Optional[MarketBar]:
        """Advance to next bar and return it."""
        if self.current_day is None:
            return None
            
        self.current_bar_idx += 1
        return self.current_day.get_bar_at_idx(self.current_bar_idx)
        
    def is_done(self) -> bool:
        """Check if we've reached end of trading day."""
        if self.current_day is None:
            return True
        return self.current_bar_idx >= len(self.current_day.bars) - 1


# Global data provider instance
_data_provider: Optional[HistoricalDataProvider] = None

# Project root detection
def _get_project_root() -> Path:
    """Find project root by looking for data directory."""
    # Start from this file's location
    current = Path(__file__).parent.resolve()
    
    # Go up until we find data directory or reach root
    for parent in [current] + list(current.parents):
        if (parent / "data" / "cache").exists():
            return parent
    
    # Fallback: assume we're in src/rl/ and project root is 2 levels up
    return current.parent.parent


def get_data_provider(
    intraday_data_dir: Optional[str] = None,
    cache_index_path: Optional[str] = None,
    date_range: Optional[Tuple[str, str]] = ("2020-01-01", "2024-12-31"),
    min_bars_per_day: int = 100
) -> HistoricalDataProvider:
    """
    Get or create global data provider instance.
    
    Args:
        intraday_data_dir: Directory with Parquet files (auto-detected if None)
        cache_index_path: Path to cache the index (auto-detected if None)
        date_range: (start_date, end_date) tuple to filter trading days
        min_bars_per_day: Minimum bars required for a valid trading day
    """
    global _data_provider
    if _data_provider is None:
        project_root = _get_project_root()
        
        # Auto-detect paths
        if intraday_data_dir is None:
            intraday_data_dir = str(project_root / "data" / "cache" / "1min_extended")
        elif not Path(intraday_data_dir).is_absolute():
            intraday_data_dir = str(project_root / intraday_data_dir)
            
        if cache_index_path is None:
            cache_index_path = str(project_root / "data" / "cache" / "trading_days_index.pkl")
        elif not Path(cache_index_path).is_absolute():
            cache_index_path = str(project_root / cache_index_path)
        
        logger.info(f"Project root: {project_root}")
        logger.info(f"Data directory: {intraday_data_dir}")
        
        _data_provider = HistoricalDataProvider(
            intraday_data_dir=intraday_data_dir,
            cache_index_path=cache_index_path,
            date_range=date_range,
            min_bars_per_day=min_bars_per_day,
            scan_parallel=True
        )
    return _data_provider


def reset_data_provider():
    """Reset global data provider."""
    global _data_provider
    _data_provider = None
