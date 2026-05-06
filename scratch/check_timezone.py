"""Check timezone handling in tick data."""
from datetime import datetime
from src.backtest.historical_tick_fetcher import tick_fetcher

date = datetime(2021, 6, 2)
tick_df = tick_fetcher.fetch_combined_tick_data('AMC', date, use_quotes=False)

if not tick_df.is_empty():
    print('Sample timestamps:')
    for row in tick_df.head(5).to_dicts():
        print(f"  {row['timestamp']}")
    
    print(f"\nTotal ticks: {len(tick_df)}")
    
    # Check hour distribution
    hours = tick_df.with_columns([
        tick_df['timestamp'].dt.hour().alias('hour')
    ]).group_by('hour').count().sort('hour')
    
    print("\nHour distribution:")
    for row in hours.to_dicts():
        print(f"  Hour {row['hour']:02d}: {row['count']:,} ticks")
