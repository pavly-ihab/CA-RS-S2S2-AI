# modules/feature_selection.py
"""
Step 3: Universal Drop Non-Interesting Features
Interactive and dynamic column selection allowing removal of uninformative IDs,
high-cardinality metadata, or sparse features for any dataset.
"""
from typing import List, Optional
import pandas as pd
from modules.interactive_cli import prompt_multi_column_selection


def select_and_drop_features(
    df: pd.DataFrame,
    target_col: str,
    suggested_drops: Optional[List[str]] = None,
    explicit_drops: Optional[List[str]] = None,
    auto_mode: bool = False
) -> pd.DataFrame:
    """
    Interactively prompts user to select columns to remove from the dataset.
    """
    if df is None or df.empty:
        return df

    df_clean = df.copy()
    available_cols = [c for c in df_clean.columns if c != target_col]

    if explicit_drops is not None:
        cols_to_drop = [c for c in explicit_drops if c in available_cols]
    else:
        # Filter suggested drops so target is never dropped
        suggested = [c for c in (suggested_drops or []) if c in available_cols]
        cols_to_drop = prompt_multi_column_selection(
            prompt_text="Select non-interesting or ID features to drop (e.g. 1, 2, 4 or 'none'):",
            columns=available_cols,
            suggested_cols=suggested,
            auto_mode=auto_mode
        )

    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)
        print(f"\n[Success] Dropped {len(cols_to_drop)} columns: {cols_to_drop}")
        print(f"Remaining Dataset Shape: {df_clean.shape}")
    else:
        print("\n[Notice] No columns dropped. Retaining all features.")

    return df_clean
