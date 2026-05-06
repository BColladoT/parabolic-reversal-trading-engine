#!/usr/bin/env python3
import polars as pl
df = pl.read_parquet("data/cache/AACG_1min_20190101_20241231.parquet")
print("Columns:", df.columns)
print("\nSample data:")
print(df.head(3))
print("\nDate range:")
print("First:", df[df.columns[0]][0])
print("Last:", df[df.columns[0]][-1])
