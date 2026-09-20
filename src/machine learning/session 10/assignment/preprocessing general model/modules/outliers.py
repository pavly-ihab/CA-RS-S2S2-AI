# modules/outliers.py
"""
Steps 8 & 9: Universal Check Outliers & Handle Outliers (Interactive)
Computes IQR bounds and Z-scores for all continuous numeric features across any dataset,
offering interactive treatment strategies (Clipping, Median, Mean, Trimming, or Keeping).
"""
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from modules.interactive_cli import prompt_user_choice


def check_outliers(
    df: pd.DataFrame,
    num_cols: Optional[List[str]] = None,
    target_col: Optional[str] = None,
    multiplier: float = 1.5,
    z_threshold: float = 3.0
) -> pd.DataFrame:
    """
    Step 8: Diagnoses outliers across all continuous numerical columns using IQR & Z-Scores.
    """
    if df is None or df.empty:
        print("[Warning] Empty DataFrame provided.")
        return pd.DataFrame()

    if num_cols is None:
        num_cols = [
            c for c in df.select_dtypes(include=["number"]).columns
            if c != target_col and df[c].nunique() > 10
        ]

    valid_cols = [c for c in num_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not valid_cols:
        print("\n--- Outlier Detection ---")
        print("No continuous numerical features found to evaluate for outliers.")
        return pd.DataFrame()

    summary_rows = []
    print("\n--- Outlier Detection Diagnostics ---")

    for col in valid_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower_fence = q1 - (multiplier * iqr)
        upper_fence = q3 + (multiplier * iqr)

        iqr_outliers = series[(series < lower_fence) | (series > upper_fence)]
        iqr_count = len(iqr_outliers)
        iqr_pct = (iqr_count / len(series)) * 100

        mean_val = float(series.mean())
        std_val = float(series.std())
        if std_val > 0:
            z_scores = np.abs((series - mean_val) / std_val)
            z_count = int((z_scores > z_threshold).sum())
        else:
            z_count = 0

        summary_rows.append({
            "Feature": col,
            "Q1": round(q1, 2),
            "Q3": round(q3, 2),
            "IQR": round(iqr, 2),
            "Lower Fence": round(lower_fence, 2),
            "Upper Fence": round(upper_fence, 2),
            "IQR Outliers": iqr_count,
            "IQR Outlier %": round(iqr_pct, 2),
            f"Z-Score (|z|>{z_threshold})": z_count
        })

    if not summary_rows:
        return pd.DataFrame()

    report = pd.DataFrame(summary_rows).set_index("Feature")
    print(report.T)
    return report


def handle_outliers_interactive(
    df: pd.DataFrame,
    num_cols: Optional[List[str]] = None,
    target_col: Optional[str] = None,
    auto_mode: bool = False
) -> pd.DataFrame:
    """
    Step 9: Interactively prompts user for treatment strategy across continuous numerical features.
    """
    if df is None or df.empty:
        return df

    if num_cols is None:
        num_cols = [
            c for c in df.select_dtypes(include=["number"]).columns
            if c != target_col and df[c].nunique() > 10
        ]

    valid_cols = [c for c in num_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not valid_cols:
        return df

    print("\n[?] Outlier Handling Strategy Configuration")
    options = {
        "1": "Clip / Winsorize to IQR Fences (Preserves rows & caps extremes)",
        "2": "Replace Outliers with Column Median",
        "3": "Replace Outliers with Column Mean",
        "4": "Drop Outlier Rows",
        "5": "Keep Raw Values (No treatment)"
    }

    choice = prompt_user_choice(
        prompt_text=f"Choose treatment for numeric outliers in {valid_cols}:",
        options=options,
        default_key="1",
        auto_mode=auto_mode
    )

    if choice == "5":
        print("[Notice] Keeping outliers unchanged.")
        return df

    df_cleaned = df.copy()
    rows_before = len(df_cleaned)

    for col in valid_cols:
        series = df_cleaned[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - (1.5 * iqr)
        upper_fence = q3 + (1.5 * iqr)

        outlier_mask = (df_cleaned[col] < lower_fence) | (df_cleaned[col] > upper_fence)
        num_outliers = int(outlier_mask.sum())

        if num_outliers == 0:
            print(f"[Notice] No outliers in '{col}'.")
            continue

        if choice == "1":
            df_cleaned[col] = df_cleaned[col].clip(lower=lower_fence, upper=upper_fence)
            print(f"[Success] Clipped {num_outliers} outliers in '{col}' to [{lower_fence:.2f}, {upper_fence:.2f}].")
        elif choice == "2":
            med_val = df_cleaned[col].median()
            df_cleaned[col] = np.where(outlier_mask, med_val, df_cleaned[col])
            print(f"[Success] Replaced {num_outliers} outliers in '{col}' with median: {med_val:.2f}")
        elif choice == "3":
            mean_val = df_cleaned[col].mean()
            df_cleaned[col] = np.where(outlier_mask, mean_val, df_cleaned[col])
            print(f"[Success] Replaced {num_outliers} outliers in '{col}' with mean: {mean_val:.2f}")
        elif choice == "4":
            df_cleaned = df_cleaned[~outlier_mask]
            print(f"[Success] Dropped {num_outliers} rows containing outliers in '{col}'.")

    if choice == "4":
        print(f"[Notice] Rows before: {rows_before} -> Rows after: {len(df_cleaned)}")

    return df_cleaned
