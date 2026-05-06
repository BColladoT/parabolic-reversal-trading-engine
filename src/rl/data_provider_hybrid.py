"""
Hybrid Data Provider for RL Training

Combines:
1. CSV setups (proven winners with actual trades)
2. Parquet data (high volatility days)

Key difference: Instead of filtering to specific hours, we filter to bars
where VWAP deviation > 20%, ensuring the agent learns in valid trading conditions.
"""

import os
import pickle
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import namedtuple

import numpy as np
import polars as pl
import yaml

# Bar data structure for environment compatibility
Bar = namedtuple('Bar', ['open', 'high', 'low', 'close', 'volume', 'vwap', 'vwap_deviation', 'timestamp'])

from src.rl.config import RL_CONFIG
from src.utils.logger import logger

# Minimum bars required for a valid trading day episode (1 hour of 1-min bars).
# Used in both index building (_load_parquet_setups) and episode loading (_load_trading_day).
MIN_EPISODE_BARS = 60


class HybridDataProvider:
    """
    Hybrid data provider that loads from both CSV setups and Parquet files.
    
    CSV setups are validated winners with actual profitable trades.
    Parquet setups add variety from all high-volatility days.
    
    CRITICAL: Supports date range filtering to prevent WFO data leakage.
    When date_range is set, only episodes within [start_date, end_date] 
    are available for sampling.
    """
    
    def __init__(
        self,
        csv_path: str = "reports/relaxed_909_backtest.csv",
        parquet_dir: str = "data/cache/1min_extended",
        cache_dir: str = None,  # Will be resolved to absolute path
        csv_weight: float = 0.7,
        min_vwap_deviation: float = 15.0,  # Lowered from 20% to capture more tradeable setups
        skip_parquet_scan: bool = False,  # Scan all Parquet files for unbiased training universe
        date_range: Optional[Tuple[Optional[str], Optional[str]]] = None,  # (start_date, end_date) for WFO
        seed: Optional[int] = None,  # Seed for reproducible episode selection
        mode: str = "train",  # "train" or "eval" - for logging/validation purposes
    ):
        """
        Initialize hybrid data provider.
        
        Args:
            csv_path: Path to CSV with backtest results
            parquet_dir: Directory with Parquet files
            cache_dir: Directory for caching index
            csv_weight: Probability of sampling from CSV (vs Parquet)
            min_vwap_deviation: Minimum VWAP deviation for valid setups
            skip_parquet_scan: Skip Parquet scanning (for quick testing with CSV only)
            date_range: Optional (start_date, end_date) tuple to filter episodes.
                       Format: "YYYY-MM-DD". Used for WFO to prevent data leakage.
            seed: Random seed for reproducible episode selection
            mode: "train" or "eval" - determines sampling behavior and validation
        """
        self.csv_path = Path(csv_path)
        self.parquet_dir = Path(parquet_dir)
        
        # Validate mode
        if mode not in ("train", "eval"):
            raise ValueError(f"mode must be 'train' or 'eval', got {mode}")
        self.mode = mode
        
        # Resolve cache_dir to absolute path
        if cache_dir is None:
            # Try to find the project root
            possible_roots = [
                Path("/mnt/c/quant_trading"),
                Path.home() / "quant_trading",
                Path.cwd(),
            ]
            for root in possible_roots:
                if (root / "src" / "scripts" / "data" / "cache").exists() or (root / "src").exists():
                    self.cache_dir = root / "src" / "scripts" / "data" / "cache"
                    break
            else:
                self.cache_dir = Path("src/scripts/data/cache")
        else:
            self.cache_dir = Path(cache_dir)
        self.csv_weight = csv_weight
        self.min_vwap_deviation = min_vwap_deviation
        self.skip_parquet_scan = skip_parquet_scan
        self.date_range = date_range
        self.seed = seed
        
        # Initialize random state for reproducibility
        self._rng = random.Random(seed)
        np.random.seed(seed)
        
        # Log initialization with mode
        logger.info(f"[{self.mode.upper()}] HybridDataProvider initialized")
        if date_range:
            logger.info(f"[{self.mode.upper()}] Date range: {date_range[0]} to {date_range[1]}")
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or build index
        self.index_path = self.cache_dir / "hybrid_index.pkl"
        self.csv_setups: List[Dict] = []
        self.parquet_setups: List[Dict] = []
        
        # Current episode state
        self.current_data: Optional[pl.DataFrame] = None
        self.current_symbol: Optional[str] = None
        self.current_date: Optional[str] = None
        self.current_bar_idx: int = 0
        self.start_bar_idx: int = 0  # First bar where VWAP > 20%
        self.current_source: str = 'unknown'
        self.current_max_vwap_dev: float = 0.0
        
        self._load_or_build_index()
    
    def _load_or_build_index(self):
        """Load cached index or build from scratch."""
        logger.info(f"Index path: {self.index_path}, exists: {self.index_path.exists()}")
        if self.index_path.exists():
            logger.info(f"Loading cached index: {self.index_path}")
            with open(self.index_path, 'rb') as f:
                index = pickle.load(f)
                all_csv_setups = index['csv_setups']
                all_parquet_setups = index['parquet_setups']
            
            # Apply date range filtering if specified
            self.csv_setups = self._filter_by_date_range(all_csv_setups)
            self.parquet_setups = self._filter_by_date_range(all_parquet_setups)
            
            logger.info(f"  - CSV setups: {len(self.csv_setups)} (filtered from {len(all_csv_setups)})")
            logger.info(f"  - Parquet setups: {len(self.parquet_setups)} (filtered from {len(all_parquet_setups)})")
            if self.date_range:
                logger.info(f"  - Date range filter: {self.date_range[0]} to {self.date_range[1]}")
        else:
            logger.info("Building index from scratch...")
            self._build_index()
    
    def _filter_by_date_range(self, setups: List[Dict]) -> List[Dict]:
        """
        Filter setups to only include those within date_range.
        
        This is CRITICAL for WFO to prevent data leakage - ensures that
        training only sees training dates and testing only sees test dates.
        """
        if self.date_range is None:
            return setups
        
        start_date, end_date = self.date_range
        filtered = []
        
        for setup in setups:
            date_str = setup.get('date', '')
            if not date_str:
                continue
            
            # Include if within range (inclusive)
            if start_date and date_str < start_date:
                continue
            if end_date and date_str > end_date:
                continue
            
            filtered.append(setup)
        
        return filtered
    
    def _build_index(self):
        """Build index of valid trading days from both sources."""
        # Load CSV setups
        if self.csv_path.exists():
            self._load_csv_setups()
        else:
            logger.warning(f"CSV file not found: {self.csv_path}")
        
        # Load Parquet setups (unless skipped)
        if self.skip_parquet_scan:
            logger.info("Skipping Parquet scan (quick test mode)")
        elif self.parquet_dir.exists():
            self._load_parquet_setups()
        else:
            logger.warning(f"Parquet directory not found: {self.parquet_dir}")
        
        # Save index
        index = {
            'csv_setups': self.csv_setups,
            'parquet_setups': self.parquet_setups,
            'built_at': datetime.now().isoformat()
        }
        with open(self.index_path, 'wb') as f:
            pickle.dump(index, f)
        
        total = len(self.csv_setups) + len(self.parquet_setups)
        logger.info(f"Index complete: {total} total setups")
    
    def _load_csv_setups(self):
        """Load setups from CSV with trade data."""
        import pandas as pd
        
        df = pd.read_csv(self.csv_path)
        logger.info(f"Loading CSV: {len(df)} rows")
        
        valid_count = 0
        for _, row in df.iterrows():
            symbol = row['symbol']
            date_str = row['date']
            total_pnl = row.get('pnl', row.get('total_pnl', 0))
            
            # Only include setups with profitable trades (>$100 to filter noise)
            if total_pnl <= 100:
                continue
            
            # Check if data file exists (handle both naming patterns)
            data_file = self.parquet_dir / f"{symbol}.parquet"
            if not data_file.exists():
                # Try extended naming pattern: SYMBOL_1min_20190101_20241231.parquet
                matching_files = list(self.parquet_dir.glob(f"{symbol}_1min_*.parquet"))
                if matching_files:
                    data_file = matching_files[0]
                else:
                    continue
            if not data_file.exists():
                continue
            
            # Validate VWAP data exists and meets threshold
            if self._validate_vwap_in_data(symbol, date_str):
                self.csv_setups.append({
                    'symbol': symbol,
                    'date': date_str,
                    'source': 'csv',
                    'pnl': total_pnl
                })
                valid_count += 1
        
        logger.info(f"  Valid CSV setups with trades: {valid_count}")
    
    def _load_parquet_setups(self):
        """Load setups from ALL Parquet files (unbiased training universe).
        
        CRITICAL: Includes ALL trading days - boring, noisy, losing, AND winning.
        The RL agent MUST learn to output 0.0 (Hold/Flat) when conditions don't align.
        """
        # Get all parquet files
        parquet_files = list(self.parquet_dir.glob("*.parquet"))
        logger.info(f"Scanning {len(parquet_files)} symbols")
        
        scanned = 0
        skipped_short_days = 0
        for pq_file in parquet_files:
            symbol = pq_file.stem
            # Handle extended naming pattern
            if '_1min_' in symbol:
                symbol = symbol.split('_1min_')[0]
            
            try:
                df = pl.read_parquet(pq_file)
                df = self._convert_to_et(df)

                # Filter to market hours (ET) and group by date
                df = df.filter(
                    (pl.col('timestamp').dt.hour() >= 9) &
                    (pl.col('timestamp').dt.hour() <= 16)
                )
                df = df.with_columns([
                    pl.col('timestamp').dt.date().alias('date')
                ])

                for date_val in df['date'].unique():
                    date_df = df.filter(pl.col('date') == date_val)
                    bar_count = len(date_df)

                    if bar_count < MIN_EPISODE_BARS:
                        skipped_short_days += 1
                        continue

                    # Calculate VWAP per-date so accumulators reset each day.
                    # This matches _load_trading_day() which also filters to
                    # a single date before calling _calculate_vwap().
                    date_df = self._calculate_vwap(date_df)
                    date_df = date_df.with_columns([
                        ((pl.col('close') - pl.col('vwap')) / pl.col('vwap') * 100).alias('vwap_dev')
                    ])
                    max_vwap_dev = date_df['vwap_dev'].abs().max()

                    # Include ALL trading days that meet the bar minimum -
                    # boring, noisy, losing, AND winning.
                    # The agent must learn when NOT to trade as much as when TO trade
                    self.parquet_setups.append({
                        'symbol': symbol,
                        'date': date_val.strftime('%Y-%m-%d'),
                        'source': 'parquet',
                        'max_vwap_dev': float(max_vwap_dev),
                        'bar_count': bar_count
                    })
                
                scanned += 1
                if scanned % 500 == 0:
                    logger.info(f"  Scanned {scanned} symbols...")
                    
            except Exception as e:
                logger.debug(f"Error scanning {symbol}: {e}")
                continue
        
        # Log distribution of volatility levels
        volatile_days = sum(1 for s in self.parquet_setups if s['max_vwap_dev'] >= self.min_vwap_deviation)
        boring_days = len(self.parquet_setups) - volatile_days
        
        logger.info(f"  Total trading days: {len(self.parquet_setups)}")
        logger.info(f"    - Volatile days (VWAP > {self.min_vwap_deviation}%): {volatile_days}")
        logger.info(f"    - Boring/noisy days (VWAP < {self.min_vwap_deviation}%): {boring_days}")
        logger.info(f"  Agent will learn to Hold/Flat on {boring_days} non-setup days")
        logger.info(f"    - Skipped (< {MIN_EPISODE_BARS} bars): {skipped_short_days} date-entries")
    
    def _validate_vwap_in_data(self, symbol: str, date_str: str) -> bool:
        """Check if VWAP data exists and meets threshold for a specific date."""
        try:
            # Handle both naming patterns
            data_file = self.parquet_dir / f"{symbol}.parquet"
            if not data_file.exists():
                matching_files = list(self.parquet_dir.glob(f"{symbol}_1min_*.parquet"))
                if matching_files:
                    data_file = matching_files[0]
                else:
                    return False
            
            if not data_file.exists():
                return False
            
            df = pl.read_parquet(data_file)
            df = self._convert_to_et(df)

            # Parse date
            date_val = datetime.strptime(date_str, '%Y-%m-%d').date()

            # Filter to date and market hours (ET)
            df = df.filter(
                (pl.col('timestamp').dt.date() == date_val) &
                (pl.col('timestamp').dt.hour() >= 9) &
                (pl.col('timestamp').dt.hour() <= 16)
            )
            
            if len(df) == 0:
                return False
            
            # Always recalculate VWAP from market open
            df = self._calculate_vwap(df)
            
            # Calculate VWAP deviation
            df = df.with_columns([
                ((pl.col('close') - pl.col('vwap')) / pl.col('vwap') * 100).alias('vwap_dev')
            ])
            
            max_dev = df['vwap_dev'].abs().max()
            return max_dev >= self.min_vwap_deviation
            
        except Exception as e:
            logger.debug(f"Error validating {symbol} {date_str}: {e}")
            return False
    
    def _convert_to_et(self, df: pl.DataFrame) -> pl.DataFrame:
        """Convert UTC timestamps to America/New_York for correct market-hours filtering.

        Parquet timestamps are UTC. All dt.hour() filtering must operate on ET hours
        to correctly capture the 9:00 AM - 4:59 PM ET session.
        Safe to call on already-ET timestamps (convert_time_zone is a no-op).
        """
        return df.with_columns(
            pl.col('timestamp').dt.convert_time_zone('America/New_York')
        )

    def _calculate_vwap(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate VWAP anchored from market open (9:30 AM ET)."""
        # Convert timestamps to ET and extract hour/minute
        et_times = df['timestamp'].dt.convert_time_zone('America/New_York')
        hours = et_times.dt.hour().cast(pl.Int32).to_numpy()
        minutes = et_times.dt.minute().cast(pl.Int32).to_numpy()
        
        # Calculate minutes from midnight (using numpy to avoid overflow)
        minutes_from_midnight = hours * 60 + minutes
        market_open_minutes = 9 * 60 + 30
        after_open_mask = minutes_from_midnight >= market_open_minutes
        
        # Calculate typical price
        df = df.with_columns([
            ((pl.col('high') + pl.col('low') + pl.col('close')) / 3).alias('typical_price')
        ])
        
        # Calculate PV (price * volume)
        typical_price = df['typical_price'].to_numpy()
        volume = df['volume'].to_numpy()
        close = df['close'].to_numpy()
        pv = typical_price * volume
        
        # Calculate cumulative VWAP from market open
        cum_pv = 0.0
        cum_vol = 0.0
        vwap_values = []
        
        for i in range(len(df)):
            if after_open_mask[i]:
                cum_pv += pv[i]
                cum_vol += volume[i]
                vwap_values.append(cum_pv / cum_vol if cum_vol > 0 else close[i])
            else:
                vwap_values.append(close[i])
        
        df = df.with_columns([
            pl.Series('vwap', vwap_values)
        ])
        
        return df.drop(['typical_price'])
    
    def _load_trading_day(self, symbol: str, date_str: str) -> Optional[pl.DataFrame]:
        """
        Load and prepare trading day data.
        
        Returns DataFrame with all bars, and finds the first bar where VWAP > 20%.
        """
        try:
            # Handle both naming patterns
            data_file = self.parquet_dir / f"{symbol}.parquet"
            if not data_file.exists():
                matching_files = list(self.parquet_dir.glob(f"{symbol}_1min_*.parquet"))
                if matching_files:
                    data_file = matching_files[0]
                else:
                    return None
            
            df = pl.read_parquet(data_file)
            df = self._convert_to_et(df)

            # Parse date
            date_val = datetime.strptime(date_str, '%Y-%m-%d').date()

            # Filter to date and market hours (ET)
            df = df.filter(
                (pl.col('timestamp').dt.date() == date_val) &
                (pl.col('timestamp').dt.hour() >= 9) &
                (pl.col('timestamp').dt.hour() <= 16)
            )
            
            if len(df) < MIN_EPISODE_BARS:
                logger.warning(f"Insufficient data for {symbol} {date_str}: {len(df)} bars")
                return None
            
            # Always recalculate VWAP from market open
            df = self._calculate_vwap(df)
            
            # Calculate VWAP deviation
            df = df.with_columns([
                ((pl.col('close') - pl.col('vwap')) / pl.col('vwap') * 100).alias('vwap_dev')
            ])
            
            # Find first bar where VWAP > 20% (entry threshold - 3% buffer)
            entry_threshold = RL_CONFIG.get('min_vwap_deviation_entry', 20.0)
            valid_bars = df.filter(pl.col('vwap_dev').abs() > (entry_threshold - 3.0))
            
            if len(valid_bars) == 0:
                logger.warning(f"No bars with VWAP > {entry_threshold - 3}% for {symbol} {date_str}")
                return None
            
            # Get the index of the first valid bar
            first_valid_idx = valid_bars.select(pl.first('__row_index__')).to_numpy()[0, 0] if '__row_index__' in valid_bars.columns else 0
            
            # Add row index if not present
            if '__row_index__' not in df.columns:
                df = df.with_row_index('__row_index__')
                valid_bars = df.filter(pl.col('vwap_dev').abs() > (entry_threshold - 3.0))
                first_valid_idx = int(valid_bars['__row_index__'][0])
            
            logger.info(f"Loaded {symbol} {date_str}: {len(df)} bars, first valid at bar {first_valid_idx}")
            return df
            
        except Exception as e:
            logger.warning(f"Failed to load {symbol} {date_str}: {e}")
            return None
    
    def reset_episode(self) -> bool:
        """
        Reset and load a new episode.
        
        Uses seeded RNG for reproducible episode selection.
        CRITICAL: Runtime assertion verifies date is within configured bounds.
        Returns True if successful, False otherwise.
        """
        max_attempts = 10
        
        for attempt in range(max_attempts):
            # Choose source based on weight (using seeded RNG)
            if self._rng.random() < self.csv_weight and len(self.csv_setups) > 0:
                setup = self._rng.choice(self.csv_setups)
            elif len(self.parquet_setups) > 0:
                setup = self._rng.choice(self.parquet_setups)
            else:
                logger.error(f"[{self.mode.upper()}] No setups available")
                return False
            
            symbol = setup['symbol']
            date_str = setup['date']
            self.current_source = setup.get('source', 'unknown')
            self.current_max_vwap_dev = setup.get('max_vwap_dev', 0.0)

            # =====================================================================
            # CRITICAL: Runtime assertion to prevent WFO data leakage
            # =====================================================================
            if self.date_range is not None:
                start_date, end_date = self.date_range
                if start_date and date_str < start_date:
                    logger.error(
                        f"[{self.mode.upper()}] DATA LEAKAGE DETECTED: "
                        f"Episode date {date_str} < {start_date} (start_date). "
                        f"Symbol: {symbol}"
                    )
                    raise RuntimeError(
                        f"WFO Data Leakage: Attempted to sample episode from {date_str} "
                        f"which is before training start {start_date}"
                    )
                if end_date and date_str > end_date:
                    logger.error(
                        f"[{self.mode.upper()}] DATA LEAKAGE DETECTED: "
                        f"Episode date {date_str} > {end_date} (end_date). "
                        f"Symbol: {symbol}"
                    )
                    raise RuntimeError(
                        f"WFO Data Leakage: Attempted to sample episode from {date_str} "
                        f"which is after training end {end_date}"
                    )
            # =====================================================================
            
            # Load data
            df = self._load_trading_day(symbol, date_str)
            if df is None:
                continue
            
            # Find first bar where VWAP > 20%
            entry_threshold = RL_CONFIG.get('min_vwap_deviation_entry', 20.0)
            valid_bars = df.filter(pl.col('vwap_dev').abs() > (entry_threshold - 3.0))
            
            if len(valid_bars) == 0:
                logger.warning(f"No valid entry bars for {symbol} {date_str}")
                continue
            
            # Get starting index
            first_valid_bar = valid_bars.row(0, named=True)
            if '__row_index__' in first_valid_bar:
                self.start_bar_idx = int(first_valid_bar['__row_index__'])
            else:
                # Find index by filtering
                first_ts = first_valid_bar['timestamp']
                all_timestamps = df['timestamp'].to_list()
                self.start_bar_idx = all_timestamps.index(first_ts)

            # FIX 3: Advance start_bar_idx to first bar at or after 09:45 ET
            # Bars before 09:45 are dead (entry window opens at 09:45)
            # BUGFIX: cast to Int32 before multiplying to avoid int8 overflow
            # (Polars dt.hour() returns Int8; 9*60=540 overflows int8 max=127)
            et_timestamps = df['timestamp'].dt.convert_time_zone('America/New_York')
            bar_minutes = (et_timestamps.dt.hour().cast(pl.Int32) * 60 + et_timestamps.dt.minute().cast(pl.Int32)).to_numpy()
            window_open_minutes = 9 * 60 + 45  # 09:45
            first_window_bar = int(np.searchsorted(bar_minutes, window_open_minutes))
            self.start_bar_idx = max(self.start_bar_idx, first_window_bar)

            if self.start_bar_idx >= len(df):
                logger.warning(f"No bars at/after 09:45 for {symbol} {date_str}")
                continue

            # Set episode state
            self.current_data = df
            self.current_symbol = symbol
            self.current_date = date_str
            self.current_bar_idx = self.start_bar_idx
            
            logger.info(f"[{self.mode.upper()}] Episode reset: {symbol} {date_str} "
                       f"(bar {self.start_bar_idx}/{len(df)}, "
                       f"VWAP dev: {first_valid_bar['vwap_dev']:.1f}%)"
                       f"{' [DATE_CHECKED]' if self.date_range else ''}")
            return True
        
        logger.error(f"[{self.mode.upper()}] Failed to load valid episode after {max_attempts} attempts")
        logger.error(f"  CSV setups: {len(self.csv_setups)}, Parquet setups: {len(self.parquet_setups)}")
        return False
    
    def get_observation(self, lookback: int = 60) -> Optional[np.ndarray]:
        """
        Get observation with lookback window (includes current bar).
        
        WARNING: This method includes the CURRENT bar. For pre-decision
        sequences (strictly before current bar), use get_pre_decision_sequence().
        
        Returns array of shape (lookback, 5) with OHLCV data.
        """
        if self.current_data is None or self.current_bar_idx < self.start_bar_idx:
            return None
        
        # Get window of data (INCLUDES current bar)
        start_idx = max(self.start_bar_idx, self.current_bar_idx - lookback + 1)
        end_idx = self.current_bar_idx + 1
        
        if end_idx > len(self.current_data):
            return None
        
        # Extract OHLCV
        window = self.current_data[start_idx:end_idx]
        
        # Need at least 1 bar
        if len(window) == 0:
            return None
        
        # Convert to numpy
        ohlcv = np.column_stack([
            window['open'].to_numpy(),
            window['high'].to_numpy(),
            window['low'].to_numpy(),
            window['close'].to_numpy(),
            window['volume'].to_numpy()
        ])
        
        # Pad with zeros if needed (NOT repeated bars - prevents leakage)
        if len(ohlcv) < lookback:
            # Use zeros for missing history - will be detected by model
            padding = np.zeros((lookback - len(ohlcv), 5), dtype=ohlcv.dtype)
            ohlcv = np.vstack([padding, ohlcv])
        
        return ohlcv
    
    def get_pre_decision_sequence(self, lookback: int = 60) -> Optional[np.ndarray]:
        """
        Get STRICTLY PRE-DECISION sequence for state encoding.
        
        CRITICAL SEMANTICS:
        - Returns exactly 'lookback' bars IMMEDIATELY PRECEDING current_bar_idx
        - Window: [current_bar_idx - lookback, current_bar_idx)
        - The LAST bar in the sequence is at index current_bar_idx - 1
        - The CURRENT bar is EXCLUDED from the sequence
        - Can access bars BEFORE start_bar_idx (earlier in the trading day)
        - Padding with zeros only for truly missing prefix bars
        
        This ensures the state sequence contains only information available
        BEFORE the current decision point, preventing future leakage.
        
        Args:
            lookback: Number of bars to return (default 60)
            
        Returns:
            np.ndarray: [lookback, 5] OHLCV array or None if error
            - Rows: [bar_t-60, bar_t-59, ..., bar_t-1] where t = current_bar_idx
            - Columns: [open, high, low, close, volume]
        """
        if self.current_data is None or self.current_bar_idx < 0:
            return None
        
        # STRICTLY PRE-DECISION: Window ends BEFORE current bar
        # Window: [current_bar_idx - lookback, current_bar_idx)
        end_idx = self.current_bar_idx  # EXCLUSIVE - current bar NOT included
        start_idx = max(0, end_idx - lookback)  # Can go back to bar 0 of day
        
        if end_idx > len(self.current_data):
            return None
        
        # Extract OHLCV (EXCLUDES current bar)
        window = self.current_data[start_idx:end_idx]
        
        if len(window) == 0:
            # No prior bars available - return all zeros
            return np.zeros((lookback, 5), dtype=np.float32)
        
        # Convert to numpy
        ohlcv = np.column_stack([
            window['open'].to_numpy(),
            window['high'].to_numpy(),
            window['low'].to_numpy(),
            window['close'].to_numpy(),
            window['volume'].to_numpy()
        ]).astype(np.float32)
        
        # Pad with zeros ONLY for truly missing prefix bars
        actual_bars = len(ohlcv)
        if actual_bars < lookback:
            padding_needed = lookback - actual_bars
            # Zeros at beginning (earliest time) - model detects missing history
            padding = np.zeros((padding_needed, 5), dtype=np.float32)
            ohlcv = np.vstack([padding, ohlcv])
        
        # Verify semantics
        assert ohlcv.shape == (lookback, 5), f"Expected ({lookback}, 5), got {ohlcv.shape}"
        assert end_idx == self.current_bar_idx, "Current bar should NOT be in sequence"
        
        return ohlcv
    
    def get_state_features(self) -> Optional[Dict[str, float]]:
        """Get additional state features for current bar."""
        if self.current_data is None or self.current_bar_idx >= len(self.current_data):
            return None
        
        row = self.current_data.row(self.current_bar_idx, named=True)
        
        return {
            'vwap_deviation': row['vwap_dev'],
            'price': row['close'],
            'volume': row['volume'],
            'bar_index': self.current_bar_idx - self.start_bar_idx,  # Relative to start
            'total_bars': len(self.current_data) - self.start_bar_idx,
        }
    
    def step(self) -> bool:
        """
        Advance to next bar.
        
        Returns True if episode continues, False if ended.
        """
        self.current_bar_idx += 1
        
        if self.current_data is None:
            return False
        
        return self.current_bar_idx < len(self.current_data)
    
    def get_total_bars(self) -> int:
        """Get total bars in current episode (from start point)."""
        if self.current_data is None:
            return 0
        return len(self.current_data) - self.start_bar_idx
    
    def get_current_bar_index(self) -> int:
        """Get current bar index (relative to start)."""
        if self.current_data is None:
            return 0
        return self.current_bar_idx - self.start_bar_idx
    
    def get_current_bar(self) -> Optional[Bar]:
        """Get current bar data as Bar namedtuple."""
        if self.current_data is None or self.current_bar_idx >= len(self.current_data):
            return None
        row = self.current_data.row(self.current_bar_idx, named=True)
        return Bar(
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume'],
            vwap=row['vwap'],
            vwap_deviation=row['vwap_dev'],
            timestamp=row['timestamp']
        )
    
    def advance(self) -> Optional[Bar]:
        """
        Advance to next bar.
        
        Returns the new bar data, or None if at end of episode.
        """
        if self.current_data is None:
            return None
        self.current_bar_idx += 1
        if self.current_bar_idx < len(self.current_data):
            return self.get_current_bar()
        return None
    
    def is_done(self) -> bool:
        """Check if episode is complete (no more bars)."""
        if self.current_data is None:
            return True
        return self.current_bar_idx >= len(self.current_data)


# Global singleton instance
_data_provider: Optional[HybridDataProvider] = None


def get_data_provider(
    csv_path: str = "reports/relaxed_909_backtest.csv",
    parquet_dir: str = "data/cache/1min_extended",
    date_range: Optional[Tuple[Optional[str], Optional[str]]] = None,
    seed: Optional[int] = None,
    mode: str = "train",
    **kwargs
) -> HybridDataProvider:
    """
    Get or create global data provider instance.
    
    CRITICAL: For WFO, always create separate instances for train/eval
    to prevent data leakage. Do NOT use singleton pattern across folds.
    
    Args:
        csv_path: Path to CSV with backtest results
        parquet_dir: Directory with Parquet files
        date_range: Optional (start_date, end_date) tuple for WFO filtering
        seed: Random seed for reproducible episode selection
        mode: "train" or "eval" - determines validation behavior
        **kwargs: Additional arguments passed to HybridDataProvider
    """
    global _data_provider
    
    # CRITICAL: For WFO safety, if date_range is specified, ALWAYS create new instance
    # This prevents cross-fold contamination via shared state
    force_new = (date_range is not None) or (seed is not None) or (mode == "eval")
    
    if _data_provider is None or force_new:
        # Try different path resolutions
        possible_csv_paths = [
            Path(csv_path),
            Path("/mnt/c/quant_trading") / csv_path,
            Path("/mnt/c/quant_trading/reports/relaxed_909_backtest.csv"),
            Path.home() / "quant_trading" / csv_path,
        ]
        
        possible_parquet_dirs = [
            Path(parquet_dir),
            Path("/mnt/c/quant_trading") / parquet_dir,
            Path("/mnt/c/quant_trading/data/cache/1min_extended"),
            Path.home() / "quant_trading" / parquet_dir,
        ]
        
        # Find existing CSV path
        actual_csv_path = None
        for p in possible_csv_paths:
            if p.exists():
                actual_csv_path = str(p)
                break
        
        # Find existing parquet dir
        actual_parquet_dir = None
        for p in possible_parquet_dirs:
            if p.exists():
                actual_parquet_dir = str(p)
                break
        
        if actual_csv_path is None:
            raise FileNotFoundError(f"CSV file not found in any location: {csv_path}")
        if actual_parquet_dir is None:
            raise FileNotFoundError(f"Parquet directory not found in any location: {parquet_dir}")
        
        logger.info(f"Using CSV: {actual_csv_path}")
        logger.info(f"Using Parquet: {actual_parquet_dir}")
        if date_range:
            logger.info(f"Date range filter: {date_range[0]} to {date_range[1]}")
        if seed:
            logger.info(f"Random seed: {seed}")
        
        provider = HybridDataProvider(
            csv_path=actual_csv_path,
            parquet_dir=actual_parquet_dir,
            date_range=date_range,
            seed=seed,
            mode=mode,
            **kwargs
        )
        
        # CRITICAL: Never cache WFO-constrained providers to prevent cross-fold leakage
        # Only cache unconstrained providers for backward compatibility
        if not force_new:
            _data_provider = provider
        else:
            logger.info(f"[{mode.upper()}] Created isolated provider (not cached due to WFO constraints)")
        
        return provider
    
    return _data_provider


def reset_data_provider():
    """Reset the global data provider (for testing)."""
    global _data_provider
    _data_provider = None


# For testing
if __name__ == "__main__":
    # Test the provider
    provider = get_data_provider()
    
    print(f"\nCSV setups: {len(provider.csv_setups)}")
    print(f"Parquet setups: {len(provider.parquet_setups)}")
    
    # Test episode reset
    for i in range(3):
        print(f"\n--- Episode {i+1} ---")
        success = provider.reset_episode()
        if success:
            print(f"Symbol: {provider.current_symbol}")
            print(f"Date: {provider.current_date}")
            print(f"Bars from start: {provider.get_total_bars()}")
            print(f"Start bar: {provider.start_bar_idx}")
            
            # Get first observation
            obs = provider.get_observation()
            if obs is not None:
                print(f"Observation shape: {obs.shape}")
                features = provider.get_state_features()
                print(f"VWAP deviation: {features['vwap_deviation']:.2f}%")
