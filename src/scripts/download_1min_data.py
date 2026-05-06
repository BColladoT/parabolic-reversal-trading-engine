#!/usr/bin/env python3
"""
Download 1-minute OHLCV data from Alpaca for RL training.

Uses the trade log to identify which symbol/date pairs need data,
then downloads full 1-minute intraday bars for each.
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Set
import polars as pl
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_setup_pairs(setups_csv: str) -> List[Tuple[str, str]]:
    """Load (symbol, date) pairs from trade log."""
    df = pl.read_csv(setups_csv)
    
    # Filter to setups with trades
    df = df.filter((pl.col("trades") > 0) & (pl.col("gain_pct") >= 60.0))
    
    pairs = []
    for row in df.iter_rows(named=True):
        symbol = row['symbol']
        date_str = row['date']
        if isinstance(date_str, str):
            pairs.append((symbol, date_str))
    
    logger.info(f"Loaded {len(pairs)} setup pairs from {setups_csv}")
    return pairs


def download_1min_bars(symbol: str, date_str: str, output_dir: Path, api_key: str, secret_key: str) -> bool:
    """Download 1-minute bars for a specific symbol and date from Alpaca."""
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        
        # Parse date
        date = datetime.strptime(date_str, "%Y-%m-%d").date()
        
        # Alpaca client
        client = StockHistoricalDataClient(api_key, secret_key)
        
        # Request 1-minute bars
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=datetime.combine(date, datetime.min.time()),
            end=datetime.combine(date, datetime.max.time()),
            feed="iex"  # Use IEX for free tier
        )
        
        bars = client.get_stock_bars(request)
        
        if symbol not in bars.data or len(bars.data[symbol]) == 0:
            logger.warning(f"No data returned for {symbol} on {date_str}")
            return False
        
        # Convert to DataFrame
        bar_list = bars.data[symbol]
        data = {
            'timestamp': [b.timestamp for b in bar_list],
            'open': [b.open for b in bar_list],
            'high': [b.high for b in bar_list],
            'low': [b.low for b in bar_list],
            'close': [b.close for b in bar_list],
            'volume': [b.volume for b in bar_list],
            'vwap': [b.vwap for b in bar_list],
            'symbol': [symbol] * len(bar_list),
        }
        
        df = pl.DataFrame(data)
        
        # Save to parquet: data/cache/1min/SYMBOL/YYYY-MM-DD.parquet
        symbol_dir = output_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = symbol_dir / f"{date_str}.parquet"
        df.write_parquet(output_file)
        
        logger.info(f"Downloaded {len(df)} 1-min bars for {symbol} {date_str} -> {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading {symbol} {date_str}: {e}")
        return False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download 1-minute Alpaca data for RL training')
    parser.add_argument('--setups-csv', type=str, default='reports/relaxed_909_backtest.csv',
                        help='Trade log CSV with symbol/date pairs')
    parser.add_argument('--output-dir', type=str, default='data/cache/1min',
                        help='Output directory for 1-minute parquet files')
    parser.add_argument('--delay', type=float, default=0.3,
                        help='Delay between API calls (seconds) to respect rate limits')
    parser.add_argument('--max-pairs', type=int, default=None,
                        help='Maximum number of pairs to download (for testing)')
    
    args = parser.parse_args()
    
    # Get API keys from environment or .env file
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET')
    
    # Try loading from .env file if not in environment
    if not api_key or not secret_key:
        env_path = Path('.env')
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        if key == 'ALPACA_API_KEY' and not api_key:
                            api_key = value
                        elif key == 'ALPACA_SECRET' and not secret_key:
                            secret_key = value
    
    if not api_key or not secret_key:
        logger.error("ALPACA_API_KEY and ALPACA_SECRET required")
        logger.error("Set them in .env file or as environment variables")
        sys.exit(1)
    
    # Load setup pairs
    pairs = load_setup_pairs(args.setups_csv)
    
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
        logger.info(f"Limited to first {args.max_pairs} pairs for testing")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track progress
    successful = 0
    failed = 0
    skipped = 0
    
    for i, (symbol, date_str) in enumerate(pairs):
        logger.info(f"[{i+1}/{len(pairs)}] Processing {symbol} {date_str}")
        
        # Check if already exists
        output_file = output_dir / symbol / f"{date_str}.parquet"
        if output_file.exists():
            logger.info(f"  Already exists, skipping")
            skipped += 1
            continue
        
        # Download
        success = download_1min_bars(symbol, date_str, output_dir, api_key, secret_key)
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Rate limiting
        time.sleep(args.delay)
    
    logger.info("="*60)
    logger.info("Download complete!")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Skipped (already exists): {skipped}")
    logger.info(f"  Total: {len(pairs)}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
