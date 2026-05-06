#!/usr/bin/env python3
import sys
sys.stderr = sys.stdout  # Redirect stderr to stdout

print("=== WSL Environment Check ===")

import os
print(f"Working dir: {os.getcwd()}")

import polars as pl
from pathlib import Path

parquet_path = Path("data/cache/AACG_1min_20190101_20241231.parquet")
print(f"\nParquet exists: {parquet_path.exists()}")

if parquet_path.exists():
    df = pl.read_parquet(parquet_path)
    print(f"Columns: {df.columns}")
    print(f"Shape: {df.shape}")
    print(f"\nFirst 3 rows:")
    print(df.head(3))
