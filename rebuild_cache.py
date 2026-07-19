"""One-off cache rebuild for hybrid_index.pkl. Delete after use."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.rl.data_provider_hybrid import HybridDataProvider

print("Rebuilding hybrid_index.pkl with full parquet scan...")
dp = HybridDataProvider(
    date_range=("2020-01-01", "2024-12-30"),
    skip_parquet_scan=False,
    mode="eval",
)
print(f"DONE. csv_setups={len(dp.csv_setups)}, parquet_setups={len(dp.parquet_setups)}")
