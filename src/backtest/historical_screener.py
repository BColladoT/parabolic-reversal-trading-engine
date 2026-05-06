"""
Historical Parabolic Screener
Scans years of historical data to find micro-cap parabolic moves.
"""
import json
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

import polars as pl
import pandas as pd
from alpaca.data.timeframe import TimeFrame

from src.utils.config import CONFIG
from src.utils.logger import logger
from src.backtest.data_fetcher import data_fetcher


@dataclass
class ParabolicSetup:
    """A detected parabolic setup for backtesting."""
    symbol: str
    date: datetime
    day_open: float
    day_high: float
    day_low: float
    day_close: float
    day_volume: int
    gain_percent: float
    market_cap: Optional[float] = None
    float_shares: Optional[float] = None
    
    # Multi-day context
    days_up: int = 1
    prior_gain_percent: float = 0.0
    avg_volume_20d: int = 0
    
    def __str__(self):
        return f"{self.symbol} {self.date.date()}: +{self.gain_percent:.1f}% " \
               f"O:${self.day_open:.2f} H:${self.day_high:.2f} V:{self.day_volume:,}"


class HistoricalParabolicScreener:
    """
    Screens historical data (2016-2024+) for parabolic micro-cap setups.
    
    Strategy Criteria:
    - Price: $1.00 - $20.00 (micro/small cap range)
    - Day gain: 50-500% (parabolic but not extreme)
    - Volume: > 3x average (unusual activity)
    - Multi-day: 2-5 consecutive green days
    - Float: Ideally < 100M shares (low float)
    """
    
    def __init__(self):
        self.cache_dir = Path("data/cache/setups")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Micro-cap universe (we'll discover these from historical data)
        self.micro_cap_symbols: Set[str] = set()
        
    def load_micro_cap_universe(
        self,
        from_file: Optional[str] = None,
        min_price: float = 0.50,
        max_price: float = 50.0,
        max_market_cap: float = 500_000_000  # $500M
    ) -> List[str]:
        """
        Load or discover micro-cap stock universe.
        
        Sources:
        1. File with pre-defined list
        2. Historical discovery (scan past data for low-priced stocks)
        """
        if from_file and Path(from_file).exists():
            with open(from_file, 'r') as f:
                symbols = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(symbols)} symbols from {from_file}")
            self.micro_cap_symbols = set(symbols)
            return symbols
        
        # Import extended universe
        from src.backtest.extended_universe import ALL_MICRO_CAP_SYMBOLS
        
        symbols = ALL_MICRO_CAP_SYMBOLS
        self.micro_cap_symbols = set(symbols)
        
        logger.info(f"Loaded default universe of {len(symbols)} micro-cap symbols")
        return symbols
    
    def scan_for_parabolic_setups(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        min_gain_percent: float = 50.0,
        max_gain_percent: float = 500.0,
        min_volume_multiplier: float = 3.0,
        use_cache: bool = True
    ) -> List[ParabolicSetup]:
        """
        Scan historical daily bars for parabolic setups.
        
        Parameters:
        -----------
        symbols : List[str]
            Universe of stocks to scan
        start_date, end_date : datetime
            Date range to scan
        min_gain_percent : float
            Minimum single-day gain (default 50%)
        max_gain_percent : float
            Maximum gain (filter out 1000%+ outliers)
        min_volume_multiplier : float
            Volume must be X times average
        """
        cache_key = f"setups_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        if use_cache and cache_path.exists():
            logger.info(f"Loading cached setups from {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        logger.info(f"Scanning {len(symbols)} symbols from {start_date.date()} to {end_date.date()}")
        
        all_setups = []
        
        for i, symbol in enumerate(symbols):
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(symbols)} symbols scanned")
            
            setups = self._scan_symbol(
                symbol, start_date, end_date,
                min_gain_percent, max_gain_percent, min_volume_multiplier
            )
            all_setups.extend(setups)
        
        # Sort by date
        all_setups.sort(key=lambda x: x.date)
        
        # Cache results
        if use_cache:
            with open(cache_path, 'wb') as f:
                pickle.dump(all_setups, f)
            logger.info(f"Cached {len(all_setups)} setups to {cache_path}")
        
        logger.info(f"Total parabolic setups found: {len(all_setups)}")
        return all_setups
    
    def _scan_symbol(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        min_gain_percent: float,
        max_gain_percent: float,
        min_volume_multiplier: float
    ) -> List[ParabolicSetup]:
        """Scan a single symbol for parabolic days."""
        setups = []
        
        try:
            # Fetch daily bars
            df = data_fetcher.fetch_alpaca_bars(
                symbol=symbol,
                start=start_date,
                end=end_date,
                timeframe=TimeFrame.Day,
                use_cache=True
            )
            
            if df.is_empty() or len(df) < 5:
                return setups
            
            # Calculate metrics
            df = df.with_columns([
                # Daily gain %
                (((pl.col('close') / pl.col('open')) - 1) * 100).alias('gain_pct'),
                # 20-day average volume
                pl.col('volume').rolling_mean(window_size=20).alias('volume_sma20'),
                # True range
                (pl.col('high') - pl.col('low')).alias('tr')
            ])
            
            # Find consecutive up days
            df = df.with_columns([
                (pl.col('close') > pl.col('open')).alias('is_green'),
            ])
            
            # Convert to pandas for easier iteration
            pdf = df.to_pandas()
            
            # Track consecutive green days
            pdf['consecutive_greens'] = 0
            consecutive = 0
            for idx in range(len(pdf)):
                if pdf.iloc[idx]['is_green']:
                    consecutive += 1
                else:
                    consecutive = 0
                pdf.at[pdf.index[idx], 'consecutive_greens'] = consecutive
            
            # Filter parabolic days
            for idx, row in pdf.iterrows():
                gain = row['gain_pct']
                volume = row['volume']
                volume_avg = row['volume_sma20']
                
                # Skip if not in gain range
                if not (min_gain_percent <= gain <= max_gain_percent):
                    continue
                
                # Skip if volume not elevated
                if volume_avg > 0 and volume < (volume_avg * min_volume_multiplier):
                    continue
                
                # Skip if price out of range
                if not (0.50 <= row['close'] <= 100.0):
                    continue
                
                # Calculate multi-day context
                days_up = int(row['consecutive_greens'])
                prior_gain = self._calculate_prior_gain(pdf, idx, days=5)
                
                setup = ParabolicSetup(
                    symbol=symbol,
                    date=row['timestamp'],
                    day_open=row['open'],
                    day_high=row['high'],
                    day_low=row['low'],
                    day_close=row['close'],
                    day_volume=int(volume),
                    gain_percent=gain,
                    days_up=days_up,
                    prior_gain_percent=prior_gain,
                    avg_volume_20d=int(volume_avg) if not pd.isna(volume_avg) else 0
                )
                
                setups.append(setup)
                
        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")
        
        return setups
    
    def _calculate_prior_gain(self, pdf: pd.DataFrame, current_idx: int, days: int = 5) -> float:
        """Calculate gain over prior N days."""
        start_idx = max(0, current_idx - days)
        if start_idx >= current_idx:
            return 0.0
        
        prior_close = pdf.iloc[start_idx]['close']
        current_open = pdf.iloc[current_idx]['open']
        
        if prior_close > 0:
            return ((current_open / prior_close) - 1) * 100
        return 0.0
    
    def filter_quality_setups(
        self,
        setups: List[ParabolicSetup],
        min_days_up: int = 2,
        max_days_up: int = 5,
        min_prior_gain: float = 30.0,
        min_volume: int = 100000,
        min_gain_percent: float = 50.0
    ) -> List[ParabolicSetup]:
        """
        Filter setups to high-quality candidates.
        
        Supports both:
        - First Red Day: 2-5 consecutive green days, 30%+ prior gain
        - Intraday Exhaustion: Single day 60%+ gain
        """
        filtered = []
        
        for setup in setups:
            # Check volume
            if setup.day_volume < min_volume:
                continue
            
            # Check price range
            if not (1.0 <= setup.day_close <= 50.0):
                continue
            
            # Check gain threshold
            if setup.gain_percent < min_gain_percent:
                continue
            
            # For multi-day setups (First Red Day)
            if setup.days_up >= 2:
                if not (min_days_up <= setup.days_up <= max_days_up):
                    continue
                if setup.prior_gain_percent < min_prior_gain:
                    continue
            
            filtered.append(setup)
        
        logger.info(f"Filtered to {len(filtered)} quality setups from {len(setups)} total")
        return filtered
    
    def export_setups_for_backtest(
        self,
        setups: List[ParabolicSetup],
        output_file: str = "reports/parabolic_setups.csv"
    ) -> str:
        """Export setups to CSV for batch backtesting."""
        Path(output_file).parent.mkdir(exist_ok=True)
        
        data = []
        for setup in setups:
            data.append({
                'symbol': setup.symbol,
                'date': setup.date.strftime('%Y-%m-%d'),
                'open': setup.day_open,
                'high': setup.day_high,
                'low': setup.day_low,
                'close': setup.day_close,
                'volume': setup.day_volume,
                'gain_percent': setup.gain_percent,
                'days_up': setup.days_up,
                'prior_gain': setup.prior_gain_percent
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        
        logger.info(f"Exported {len(data)} setups to {output_file}")
        return output_file
    
    def analyze_setup_distribution(self, setups: List[ParabolicSetup]) -> Dict:
        """Analyze distribution of setups for strategy insights."""
        if not setups:
            return {}
        
        gains = [s.gain_percent for s in setups]
        volumes = [s.day_volume for s in setups]
        days_up = [s.days_up for s in setups]
        
        # Group by symbol frequency
        symbol_counts = defaultdict(int)
        for setup in setups:
            symbol_counts[setup.symbol] += 1
        
        top_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_setups': len(setups),
            'avg_gain_percent': sum(gains) / len(gains),
            'median_gain_percent': sorted(gains)[len(gains)//2],
            'avg_volume': sum(volumes) / len(volumes),
            'avg_days_up': sum(days_up) / len(days_up),
            'top_symbols': top_symbols,
            'gain_range': (min(gains), max(gains))
        }


# Singleton
historical_screener = HistoricalParabolicScreener()
