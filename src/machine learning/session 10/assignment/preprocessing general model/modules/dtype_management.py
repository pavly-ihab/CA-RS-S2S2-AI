# modules/dtype_management.py
"""
Steps 4 & 5: Universal Check Data Types & Handle Data Types
Inspects column data types and safely converts discrete features into pandas 'category'
for any dataset.
"""
from typing import List, Optional
import pandas as pd
from modules.interactive_cli import prompt_multi_column_selection


def check_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 4: Generates a concise diagnostic report summarizing dtypes and unique values.
    """
    if df is None or df.empty:
        print("[Warning] DataFrame is empty or None.")
        return pd.DataFrame()

    report = pd.DataFrame({
        "Current Dtype": df.dtypes.astype(str),
        "Unique Count": df.nunique(),
        "Sample Value": [df[c].dropna().iloc[0] if df[c].dropna().shape[0] > 0 else None for c in df.columns]
    })

    print("\n--- Data Types Diagnostic Report ---")
    print(report.T)
    return report


def handle_dtypes_interactive(
    df: pd.DataFrame,
    target_col: str,
    suggested_cats: Optional[List[str]] = None,
    explicit_cats: Optional[List[str]] = None,
    auto_mode: bool = False
) -> pd.DataFrame:
    """
    Step 5: Safely casts chosen discrete features to pandas 'category' dtype.
    """
    if df is None or df.empty:
        return df

    df_cast = df.copy()
    available_cols = [c for c in df_cast.columns if c != target_col]

    if explicit_cats is not None:
        cat_cols = [c for c in explicit_cats if c in available_cols]
    else:
        # Prompt user with suggested categorical columns
        cat_cols = prompt_multi_column_selection(
            prompt_text="Select discrete features to convert to 'category' dtype (or 'none'):",
            columns=available_cols,
            suggested_cols=suggested_cats or [],
            auto_mode=auto_mode
        )

    if cat_cols:
        df_cast[cat_cols] = df_cast[cat_cols].astype("category")
        print(f"\n[Success] Converted {len(cat_cols)} columns to 'category': {cat_cols}")
    else:
        print("\n[Notice] No additional columns converted to categorical.")

    return df_cast
