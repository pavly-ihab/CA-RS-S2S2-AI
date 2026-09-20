# preprocessing.py
import pandas as pd
import os

def read_data_file(file_path: str) -> pd.DataFrame:
    """
    Reads a CSV file into a pandas DataFrame with robust error handling.
    """
    if not isinstance(file_path, str) or not file_path.strip():
        print(f"[Error] Invalid file path provided: '{file_path}'")
        return None

    if not os.path.exists(file_path):
        print(f"[Error] File does not exist at: '{file_path}'")
        return None

    try:
        df = pd.read_csv(file_path)
        print(f"[Success] Successfully loaded file from '{file_path}' with shape {df.shape}.")
        return df
    except pd.errors.EmptyDataError:
        print(f"[Error] The file at '{file_path}' is empty.")
    except pd.errors.ParserError:
        print(f"[Error] Could not parse '{file_path}'. Verify CSV formatting.")
    except Exception as e:
        print(f"[Error] An unexpected error occurred while reading '{file_path}': {e}")
        
    return None


def drop_unnecessary_features(df: pd.DataFrame, cols_to_drop: list) -> pd.DataFrame:
    """
    Removes specified columns dynamically without hardcoding column names.
    """
    if df is None:
        print("[Warning] DataFrame is None. Skipping column drop.")
        return None

    # Filter out columns that actually exist in the DataFrame
    existing_cols = [col for col in cols_to_drop if col in df.columns]
    missing_cols = [col for col in cols_to_drop if col not in df.columns]

    if missing_cols:
        print(f"[Notice] Columns not found in DataFrame (skipped): {missing_cols}")

    df_cleaned = df.drop(columns=existing_cols)
    print(f"[Success] Dropped {len(existing_cols)} columns. New shape: {df_cleaned.shape}")
    return df_cleaned


def check_data_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a concise data quality report:
    - Data type
    - Number of unique values
    - Null count
    Returns the transposed DataFrame for easy scanning.
    """
    if df is None or df.empty:
        print("[Warning] Empty or invalid DataFrame provided.")
        return pd.DataFrame()

    summary = pd.DataFrame({
        "Data Type": df.dtypes,
        "Unique Values": df.nunique(),
        "Missing Values": df.isnull().sum()
    })

    # Transpose so columns become rows for easy comparison
    return summary.T

# preprocessing.py (add this function)

def convert_to_categorical(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """
    Converts specified columns to pandas 'category' dtype.
    Safely ignores columns that do not exist in the DataFrame.
    """
    if df is None or df.empty:
        print("[Warning] DataFrame is None or empty. Skipping category conversion.")
        return df

    # Work on a copy or modify directly
    df = df.copy()

    # Identify existing and missing target columns
    valid_cols = [col for col in cat_cols if col in df.columns]
    missing_cols = [col for col in cat_cols if col not in df.columns]

    if missing_cols:
        print(f"[Notice] Columns not found for category conversion: {missing_cols}")

    if valid_cols:
        df[valid_cols] = df[valid_cols].astype("category")
        print(f"[Success] Converted {len(valid_cols)} columns to 'category' dtype.")

    return df