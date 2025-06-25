"""
Shared ETL step:
    1. Read 15-min CSV from data/raw
    2. Resample to hourly energy (sum)
    3. Save to data/processed/hourly.parquet
Run once:

    python -m src.preprocessing
"""

import pandas as pd
from pathlib import Path
from .config import (
    RAW_CSV_FILES,
    PROCESSED_FILE,
    DATA_PROCESSED_DIR,
    POWER_COLUMN,
    TIMESTAMP_COLUMN,
)

DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def build_dataset() -> pd.DataFrame:
    print(f"📁 Combining {len(RAW_CSV_FILES)} CSV files...")
    
    # Read and combine all CSV files
    all_dfs = []
    for i, csv_file in enumerate(RAW_CSV_FILES, 1):
        print(f"   Reading file {i}/{len(RAW_CSV_FILES)}: {csv_file.name}")
        df_temp = pd.read_csv(csv_file, parse_dates=[TIMESTAMP_COLUMN], low_memory=False)
        all_dfs.append(df_temp)
        print(f"   - Loaded {len(df_temp):,} rows from {df_temp[TIMESTAMP_COLUMN].min()} to {df_temp[TIMESTAMP_COLUMN].max()}")
    
    # Combine all DataFrames
    print("🔄 Combining and processing data...")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"   Total combined rows: {len(combined_df):,}")
    
    # Remove duplicates and sort by timestamp
    combined_df = combined_df.drop_duplicates(subset=[TIMESTAMP_COLUMN]).sort_values(TIMESTAMP_COLUMN)
    print(f"   After deduplication: {len(combined_df):,} rows")
    
    # Process to hourly data
    df15 = (
        combined_df
        .rename(columns={TIMESTAMP_COLUMN: "ds", POWER_COLUMN: "y"})
        .set_index("ds")["y"]
        .resample("1h")
        .sum(min_count=1)  # NaN if any of the four quarters is missing
        .interpolate(limit=2)  # simple gap fill (max 2 h)
        .reset_index()
    )
    
    df15.insert(0, "unique_id", "series")
    print(f"📊 Final dataset: {len(df15):,} hourly records from {df15['ds'].min()} to {df15['ds'].max()}")
    
    df15.to_parquet(PROCESSED_FILE, index=False)
    print(f"💾 Saved hourly data ➜ {PROCESSED_FILE}")
    return df15


if __name__ == "__main__":
    build_dataset()
