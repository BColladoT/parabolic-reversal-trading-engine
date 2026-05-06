#!/usr/bin/env python3
"""
Download 1-minute data for entire extended universe efficiently.

Instead of downloading day-by-day (3,527 stocks × 1,500 days = 5.3M API calls),
this downloads multi-year chunks per symbol (~3,527 calls).

Strategy:
- Download 2019-01-01 to 2024-12-31 for each symbol (~6 years)
- Store as SYMBOL_1min_20190101_20241231.parquet
- Use pagination for large datasets
- Rate limit: 200 req/min = 3.33 req/sec
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
import polars as pl
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Alpaca has a 10,000 bar limit per request for 1-min data
# 6 years × 252 days × 390 bars ≈ 590,000 bars per symbol
# Need ~60 requests per symbol


def load_extended_universe(py_path: str) -> List[str]:
    """Load list of symbols from extended universe Python module."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("extended_universe", py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    symbols = getattr(module, 'EXTENDED_MICRO_CAP_SYMBOLS', [])
    return [s.upper().strip() for s in symbols if s]


def download_symbol_1min(symbol: str, start_date: str, end_date: str, 
                         output_dir: Path, api_key: str, secret_key: str) -> bool:
    """
    Download multi-year 1-minute data for a symbol.
    
    Alpaca has 10,000 bar limit per request, so we paginate through time.
    """
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        
        client = StockHistoricalDataClient(api_key, secret_key)
        
        all_bars = []
        current_start = datetime.strptime(start_date, "%Y-%m-%d")
        final_end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Alpaca limit: 10,000 bars per request
        # 1-min bars ≈ 390 per day (market hours)
        # 10,000 / 390 ≈ 25 days per request
        chunk_days = 20  # Conservative: ~7,800 bars per request
        
        chunk_count = 0
        while current_start < final_end:
            chunk_end = min(current_start + timedelta(days=chunk_days), final_end)
            
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=current_start,
                end=chunk_end,
                feed="iex"
            )
            
            try:
                bars = client.get_stock_bars(request)
                
                if symbol in bars.data:
                    bar_list = bars.data[symbol]
                    all_bars.extend(bar_list)
                    logger.debug(f"  Chunk {chunk_count}: {len(bar_list)} bars from {current_start.date()} to {chunk_end.date()}")
                else:
                    logger.debug(f"  Chunk {chunk_count}: No data")
                    
            except Exception as e:
                logger.warning(f"  Chunk error: {e}")
            
            current_start = chunk_end
            chunk_count += 1
            
            # Rate limiting between chunks
            time.sleep(0.3)
        
        if not all_bars:
            logger.warning(f"No data returned for {symbol}")
            return False
        
        # Convert to DataFrame
        data = {
            'timestamp': [b.timestamp for b in all_bars],
            'open': [b.open for b in all_bars],
            'high': [b.high for b in all_bars],
            'low': [b.low for b in all_bars],
            'close': [b.close for b in all_bars],
            'volume': [b.volume for b in all_bars],
            'vwap': [b.vwap for b in all_bars],
            'symbol': [symbol] * len(all_bars),
        }
        
        df = pl.DataFrame(data)
        
        # Save to parquet
        output_file = output_dir / f"{symbol}_1min_{start_date.replace('-', '')}_{end_date.replace('-', '')}.parquet"
        df.write_parquet(output_file)
        
        # Calculate date range
        timestamps = df['timestamp'].to_list()
        min_date = min(timestamps).strftime('%Y-%m-%d')
        max_date = max(timestamps).strftime('%Y-%m-%d')
        unique_days = df.with_columns([
            pl.col('timestamp').dt.date().alias('date')
        ])['date'].n_unique()
        
        logger.info(f"✓ {symbol}: {len(df):,} bars, {unique_days} trading days ({min_date} to {max_date})")
        return True
        
    except Exception as e:
        logger.error(f"✗ {symbol}: {e}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Download 1-minute data for extended universe')
    parser.add_argument('--universe-py', type=str, default='src/backtest/extended_universe.py',
                        help='Extended universe CSV with symbol list')
    parser.add_argument('--output-dir', type=str, default='data/cache/1min_extended',
                        help='Output directory for parquet files')
    parser.add_argument('--start-date', type=str, default='2019-01-01',
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-12-31',
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Delay between symbols (seconds)')
    parser.add_argument('--max-symbols', type=int, default=None,
                        help='Maximum symbols to download (for testing)')
    parser.add_argument('--resume', action='store_true',
                        help='Skip symbols that already have files')
    
    args = parser.parse_args()
    
    # Load API credentials
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET')
    
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
        logger.error("ALPACA_API_KEY and ALPACA_SECRET required in .env or environment")
        sys.exit(1)
    
    # Load symbols
    symbols = load_extended_universe(args.universe_py)
    logger.info(f"Loaded {len(symbols)} symbols from {args.universe_py}")
    
    if args.max_symbols:
        symbols = symbols[:args.max_symbols]
        logger.info(f"Limited to first {args.max_symbols} symbols for testing")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate estimated time
    avg_chunks_per_symbol = 100  # ~6 years / 20 days per chunk
    time_per_symbol = avg_chunks_per_symbol * 0.3 + args.delay  # 0.3s per chunk + delay
    total_time_hours = (len(symbols) * time_per_symbol) / 3600
    
    logger.info("="*60)
    logger.info(f"Download Plan:")
    logger.info(f"  Symbols: {len(symbols)}")
    logger.info(f"  Date range: {args.start_date} to {args.end_date}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Est. time: {total_time_hours:.1f} hours ({total_time_hours/24:.1f} days)")
    logger.info(f"  Est. with 50% failures: {total_time_hours*2:.1f} hours")
    logger.info("="*60)
    
    # Download loop
    successful = 0
    failed = 0
    skipped = 0
    
    for i, symbol in enumerate(symbols):
        logger.info(f"[{i+1}/{len(symbols)}] Processing {symbol}...")
        
        # Check if already exists
        if args.resume:
            pattern = f"{symbol}_1min_*.parquet"
            existing = list(output_dir.glob(pattern))
            if existing:
                logger.info(f"  Already exists: {existing[0].name}, skipping")
                skipped += 1
                continue
        
        # Download
        success = download_symbol_1min(
            symbol, args.start_date, args.end_date,
            output_dir, api_key, secret_key
        )
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Delay between symbols
        if i < len(symbols) - 1:  # Don't delay after last symbol
            time.sleep(args.delay)
        
        # Progress report every 10 symbols
        if (i + 1) % 10 == 0:
            elapsed = (i + 1) * time_per_symbol / 60
            logger.info(f"--- Progress: {i+1}/{len(symbols)} symbols, ~{elapsed:.1f} min elapsed ---")
    
    logger.info("="*60)
    logger.info("Download complete!")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Skipped: {skipped}")
    logger.info(f"  Total: {len(symbols)}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
