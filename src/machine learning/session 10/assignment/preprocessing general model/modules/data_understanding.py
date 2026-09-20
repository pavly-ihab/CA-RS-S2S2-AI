# modules/data_understanding.py
"""
Step 2: Universal Data Understanding
Handles multi-format tabular ingestion (.csv, .tsv, .xlsx, .parquet)
and automated column profiling (ID candidates, numerical, categorical, sparse).
"""
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd


def load_dataset(file_path: Path) -> Optional[pd.DataFrame]:
    """
    Ingests tabular data across multiple common formats.
    """
    path_obj = Path(file_path).resolve()
    if not path_obj.exists():
        print(f"[Error] Dataset file not found at: {path_obj}")
        return None

    ext = path_obj.suffix.lower()
    try:
        if ext in [".csv", ".txt"]:
            # Try comma, if 1 column try semicolon or tab
            df = pd.read_csv(path_obj)
            if df.shape[1] == 1:
                df_tab = pd.read_csv(path_obj, sep="\t")
                if df_tab.shape[1] > 1:
                    df = df_tab
                else:
                    df_semi = pd.read_csv(path_obj, sep=";")
                    if df_semi.shape[1] > 1:
                        df = df_semi
        elif ext == ".tsv":
            df = pd.read_csv(path_obj, sep="\t")
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(path_obj)
        elif ext == ".parquet":
            df = pd.read_parquet(path_obj)
        else:
            # Fallback to read_csv
            df = pd.read_csv(path_obj)

        print(f"[Success] Loaded '{path_obj.name}' ({df.shape[0]} rows × {df.shape[1]} columns).")
        return df
    except Exception as exc:
        print(f"[Error] Failed to read '{path_obj.name}': {exc}")
        return None


def profile_dataset(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Automated feature categorizer identifying:
    - candidate_ids: Features with 100% uniqueness or ID naming patterns
    - high_nulls: Features with > 60% missing values
    - categorical_cols: Non-numeric or low-cardinality discrete columns
    - numerical_cols: Continuous or high-cardinality numeric features
    """
    if df is None or df.empty:
        return {}

    n_rows = len(df)
    candidate_ids = []
    high_nulls = []
    categorical_cols = []
    numerical_cols = []

    for col in df.columns:
        if col == target_col:
            continue

        series = df[col]
        nunique = series.nunique(dropna=True)
        null_pct = (series.isnull().sum() / n_rows) * 100

        # Check high nulls
        if null_pct > 60.0:
            high_nulls.append(col)

        # Check ID candidate
        col_lower = col.lower()
        is_id_name = any(col_lower.endswith(k) or col_lower == k for k in ["id", "_id", "guid", "uuid", "key"])
        if (nunique == n_rows or (is_id_name and nunique > 0.8 * n_rows)):
            candidate_ids.append(col)
            continue

        # Check dtype & cardinality
        if pd.api.types.is_numeric_dtype(series):
            if nunique <= 10 and not pd.api.types.is_float_dtype(series):
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        else:
            # Object, category, string
            if nunique < 50:
                categorical_cols.append(col)
            else:
                # High cardinality text (potential candidate to drop)
                candidate_ids.append(col)

    return {
        "candidate_ids": candidate_ids,
        "high_nulls": high_nulls,
        "categorical_cols": categorical_cols,
        "numerical_cols": numerical_cols
    }


def inspect_data(df: pd.DataFrame) -> None:
    """
    Prints key structural insights about the dataset.
    """
    if df is None or df.empty:
        print("[Warning] No data available to inspect.")
        return

    print("\n--- Dataset Dimensions ---")
    print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")

    print("\n--- Memory Footprint ---")
    mem_usage = df.memory_usage(deep=True).sum() / 1024
    print(f"Total Memory: {mem_usage:.2f} KB")

    print("\n--- First 5 Rows ---")
    print(df.head())

    print("\n--- Feature Summary Statistics ---")
    num_df = df.select_dtypes(include=["number"])
    if not num_df.empty:
        print(num_df.describe().T[["count", "mean", "std", "min", "50%", "max"]])
    else:
        print(df.describe(include="all").T)
