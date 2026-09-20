# modules/duplicates.py
"""
Steps 10 & 11: Check Duplications & Handle Duplications
Identifies duplicate records and safely removes redundancies.
"""
from typing import Optional
import pandas as pd


def check_duplicates(df: pd.DataFrame) -> int:
    """
    Step 10: Inspects the DataFrame for exact duplicate passenger rows.
    """
    if df is None or df.empty:
        print("[Warning] DataFrame is empty.")
        return 0

    num_duplicates = df.duplicated().sum()
    print("\n--- Duplication Inspection ---")
    print(f"Total Duplicate Rows Found: {num_duplicates}")

    if num_duplicates > 0:
        sample_dups = df[df.duplicated(keep=False)].head(6)
        print("Sample Duplicated Records:")
        print(sample_dups)

    return int(num_duplicates)


def handle_duplicates(df: pd.DataFrame, keep: str = "first") -> pd.DataFrame:
    """
    Step 11: Removes duplicate rows while preserving the specified occurrence.
    """
    if df is None or df.empty:
        return df

    num_duplicates = df.duplicated().sum()
    if num_duplicates == 0:
        print("[Notice] No duplicate records to remove.")
        return df

    df_dedup = df.drop_duplicates(keep=keep)
    print(f"[Success] Removed {num_duplicates} duplicate rows (keep='{keep}').")
    print(f"New Shape: {df_dedup.shape}")
    return df_dedup
