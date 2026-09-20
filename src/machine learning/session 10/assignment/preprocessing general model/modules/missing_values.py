# modules/missing_values.py
"""
Steps 6 & 7: Universal Check Nulls & Handle Nulls (Interactive)
Provides missing value diagnostics and interactive imputation strategies
applicable to any numerical or categorical feature across any dataset.
"""
from typing import Dict, Optional, List
import pandas as pd
from modules.interactive_cli import prompt_user_choice


def check_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 6: Computes missing count and percentage for every feature in any dataset.
    """
    if df is None or df.empty:
        print("[Warning] DataFrame is empty.")
        return pd.DataFrame()

    null_counts = df.isnull().sum()
    null_percent = (null_counts / len(df)) * 100

    report = pd.DataFrame({
        "Missing Count": null_counts,
        "Missing %": null_percent.round(2)
    })

    report = report.sort_values(by="Missing Count", ascending=False)
    print("\n--- Missing Values Summary ---")
    cols_with_nulls = report[report["Missing Count"] > 0]
    if cols_with_nulls.empty:
        print("No missing values detected across all features!")
    else:
        print(cols_with_nulls)

    return report


def handle_nulls_interactive(df: pd.DataFrame, auto_mode: bool = False) -> pd.DataFrame:
    """
    Step 7: Interactively guides the user through missing value treatment for each column with nulls.
    """
    if df is None or df.empty:
        return df

    df_imputed = df.copy()
    cols_with_nulls = df_imputed.columns[df_imputed.isnull().any()].tolist()

    if not cols_with_nulls:
        print("[Notice] No missing values require handling.")
        return df_imputed

    print(f"\n[?] Columns requiring missing value treatment: {cols_with_nulls}")

    for col in cols_with_nulls:
        num_nulls = df_imputed[col].isnull().sum()
        pct_nulls = (num_nulls / len(df_imputed)) * 100
        is_numeric = pd.api.types.is_numeric_dtype(df_imputed[col])

        print(f"\nFeature '{col}': {num_nulls} missing entries ({pct_nulls:.2f}%)")

        if is_numeric:
            # Check if dataset has Pclass and Sex for Age grouped median
            has_groups = (col == "Age" and "Pclass" in df_imputed.columns and "Sex" in df_imputed.columns)

            options = {}
            if has_groups:
                options["1"] = "Grouped Median (by Pclass & Sex - Advanced)"
                options["2"] = "Global Median"
                options["3"] = "Global Mean"
                options["4"] = "Impute with Constant (0)"
                options["5"] = "Drop rows with missing values"
                default_k = "1"
            else:
                options["1"] = "Impute with Median (Recommended for skewed numeric data)"
                options["2"] = "Impute with Mean"
                options["3"] = "Impute with Constant (0)"
                options["4"] = "Drop rows with missing values"
                default_k = "1"

            choice = prompt_user_choice(
                prompt_text=f"Select missing value treatment for numeric feature '{col}':",
                options=options,
                default_key=default_k,
                auto_mode=auto_mode
            )

            if has_groups and choice == "1":
                grouped_median = df_imputed.groupby(["Pclass", "Sex"], observed=False)["Age"].transform("median")
                df_imputed["Age"] = df_imputed["Age"].fillna(grouped_median)
                df_imputed["Age"] = df_imputed["Age"].fillna(df_imputed["Age"].median())
                print("[Success] Imputed 'Age' using grouped median (Pclass × Sex).")
            elif (has_groups and choice == "2") or (not has_groups and choice == "1"):
                med_val = df_imputed[col].median()
                df_imputed[col] = df_imputed[col].fillna(med_val)
                print(f"[Success] Imputed '{col}' with median: {med_val}")
            elif (has_groups and choice == "3") or (not has_groups and choice == "2"):
                mean_val = df_imputed[col].mean()
                df_imputed[col] = df_imputed[col].fillna(mean_val)
                print(f"[Success] Imputed '{col}' with mean: {mean_val:.2f}")
            elif (has_groups and choice == "4") or (not has_groups and choice == "3"):
                df_imputed[col] = df_imputed[col].fillna(0)
                print(f"[Success] Imputed '{col}' with 0")
            elif (has_groups and choice == "5") or (not has_groups and choice == "4"):
                df_imputed = df_imputed.dropna(subset=[col])
                print(f"[Success] Dropped rows missing '{col}'. New shape: {df_imputed.shape}")

        else:
            # Categorical feature
            options = {
                "1": "Impute with Mode (Most frequent category)",
                "2": "Impute with 'Unknown' category placeholder",
                "3": "Drop rows with missing values"
            }
            # If tiny fraction missing (<1%), dropping or mode are great defaults
            default_k = "1" if pct_nulls < 2.0 else "2"

            choice = prompt_user_choice(
                prompt_text=f"Select missing value treatment for categorical feature '{col}':",
                options=options,
                default_key=default_k,
                auto_mode=auto_mode
            )

            if choice == "1":
                modes = df_imputed[col].mode()
                mode_val = modes[0] if not modes.empty else "Missing"
                df_imputed[col] = df_imputed[col].fillna(mode_val)
                print(f"[Success] Imputed '{col}' with mode: '{mode_val}'")
            elif choice == "2":
                if str(df_imputed[col].dtype) == "category":
                    if "Unknown" not in df_imputed[col].cat.categories:
                        df_imputed[col] = df_imputed[col].cat.add_categories("Unknown")
                df_imputed[col] = df_imputed[col].fillna("Unknown")
                print(f"[Success] Imputed '{col}' with 'Unknown'")
            elif choice == "3":
                df_imputed = df_imputed.dropna(subset=[col])
                print(f"[Success] Dropped rows missing '{col}'. New shape: {df_imputed.shape}")

    total_remaining = df_imputed.isnull().sum().sum()
    print(f"\n[Verification] Total remaining null entries: {total_remaining}")
    return df_imputed
