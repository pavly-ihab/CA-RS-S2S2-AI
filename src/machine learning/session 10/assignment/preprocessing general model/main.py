# main.py
"""
Universal Tabular Preprocessing Framework
Master Pipeline Orchestrator executing all 15 industry-standard preprocessing steps
for ANY tabular dataset (CSV, TSV, Excel) with interactive decision gates
and Plotly visual reporting.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional
import pandas as pd

# Ensure package root is in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from config.config import (
    DEFAULT_RAW_DATA_PATH,
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    DEFAULT_TEST_SIZE,
    DEFAULT_RANDOM_STATE
)
from modules.interactive_cli import (
    print_step_header,
    prompt_dataset_path,
    prompt_column_selection,
    prompt_user_choice
)
from modules.business_understanding import print_business_understanding
from modules.data_understanding import load_dataset, inspect_data, profile_dataset
from modules.feature_selection import select_and_drop_features
from modules.dtype_management import check_dtypes, handle_dtypes_interactive
from modules.missing_values import check_nulls, handle_nulls_interactive
from modules.outliers import check_outliers, handle_outliers_interactive
from modules.duplicates import check_duplicates, handle_duplicates
from modules.visualizer import generate_visualizations
from modules.data_splitting import split_features_target, split_train_test_interactive
from modules.scaling import scale_numerical_features_interactive
from modules.encoding import encode_categorical_features_interactive


def infer_task_type(series) -> str:
    """Infers whether a target column is classification or regression."""
    if pd.api.types.is_numeric_dtype(series):
        if series.nunique() <= 10 or not pd.api.types.is_float_dtype(series):
            return "classification"
        return "regression"
    return "classification"


def run_pipeline(
    data_path: Optional[str] = None,
    target_col: Optional[str] = None,
    task_type: Optional[str] = None,
    auto_mode: bool = False,
    skip_plots: bool = False,
    auto_open: bool = False
) -> None:
    """
    Executes the universal 15-step preprocessing sequence on any dataset.
    """
    print("\n" + "#" * 70)
    print(" UNIVERSAL DATA SCIENCE PIPELINE: 15 PREPROCESSING STEPS ".center(70, "#"))
    print(f" Mode: {'AUTOMATED (Defaults)' if auto_mode else 'INTERACTIVE (User Guided)'}".center(70, " "))
    print("#" * 70)

    # -------------------------------------------------------------
    # Ingestion & Dataset Resolution
    # -------------------------------------------------------------
    if data_path:
        dataset_path = Path(data_path).expanduser().resolve()
    else:
        dataset_path = prompt_dataset_path(default_path=DEFAULT_RAW_DATA_PATH, auto_mode=auto_mode)

    dataset_name = dataset_path.stem
    print(f"\n[Dataset] Target Dataset: {dataset_path.name}")

    df = load_dataset(dataset_path)
    if df is None or df.empty:
        print("[Pipeline Terminated] Cannot proceed without valid tabular data.")
        return

    # Dataset output directories
    dataset_processed_dir = PROCESSED_DATA_DIR / dataset_name
    dataset_figures_dir = FIGURES_DIR / dataset_name
    dataset_processed_dir.mkdir(parents=True, exist_ok=True)
    dataset_figures_dir.mkdir(parents=True, exist_ok=True)

    # Target Column Resolution
    columns_list = df.columns.tolist()
    if target_col is None or target_col not in columns_list:
        # Default to 'Survived' for Titanic, or the last column
        def_target = "Survived" if "Survived" in columns_list else columns_list[-1]
        target_col = prompt_column_selection(
            prompt_text="Select target variable to predict:",
            columns=columns_list,
            default_col=def_target,
            auto_mode=auto_mode
        )

    # Task Type Resolution (Classification vs Regression)
    if task_type is None or task_type == "auto":
        inferred = "classification" if (df[target_col].nunique() <= 10 or not pd.api.types.is_numeric_dtype(df[target_col])) else "regression"
        if auto_mode:
            task_type = inferred
            print(f"[Task] Inferred Problem Type: {task_type.upper()}")
        else:
            task_choice = prompt_user_choice(
                prompt_text=f"Select problem type for target '{target_col}' (Inferred: {inferred}):",
                options={
                    "1": "Classification (Discrete categories/classes)",
                    "2": "Regression (Continuous numeric target)"
                },
                default_key="1" if inferred == "classification" else "2",
                auto_mode=auto_mode
            )
            task_type = "classification" if task_choice == "1" else "regression"
    else:
        task_type = task_type.lower()

    # -------------------------------------------------------------
    # Step 1: Business Understanding
    # -------------------------------------------------------------
    print_step_header(1, "Business Understanding")
    print_business_understanding(
        target_col=target_col,
        task_type=task_type,
        dataset_name=dataset_name,
        df=df
    )

    # -------------------------------------------------------------
    # Step 2: Data Understanding
    # -------------------------------------------------------------
    print_step_header(2, "Data Understanding")
    inspect_data(df)
    profile = profile_dataset(df, target_col=target_col)
    print("\n--- Automated Column Profiling ---")
    print(f"  - Candidate ID / Metadata Columns : {profile['candidate_ids']}")
    print(f"  - High Sparsity Columns (>60% null): {profile['high_nulls']}")
    print(f"  - Inferred Categorical Columns    : {profile['categorical_cols']}")
    print(f"  - Inferred Numerical Columns      : {profile['numerical_cols']}")

    # -------------------------------------------------------------
    # Step 3: Drop Non-Interesting Features
    # -------------------------------------------------------------
    print_step_header(3, "Drop Non-Interesting Features")
    suggested_drops = list(dict.fromkeys(profile["candidate_ids"] + profile["high_nulls"]))
    df = select_and_drop_features(
        df=df,
        target_col=target_col,
        suggested_drops=suggested_drops,
        auto_mode=auto_mode
    )

    # -------------------------------------------------------------
    # Step 4: Check Dtypes
    # -------------------------------------------------------------
    print_step_header(4, "Check Data Types")
    check_dtypes(df)

    # -------------------------------------------------------------
    # Step 5: Handle Dtypes
    # -------------------------------------------------------------
    print_step_header(5, "Handle Data Types")
    remaining_cats = [c for c in profile["categorical_cols"] if c in df.columns and c != target_col]
    df = handle_dtypes_interactive(
        df=df,
        target_col=target_col,
        suggested_cats=remaining_cats,
        auto_mode=auto_mode
    )

    # -------------------------------------------------------------
    # Step 6: Check Nulls
    # -------------------------------------------------------------
    print_step_header(6, "Check Nulls")
    check_nulls(df)

    # -------------------------------------------------------------
    # Step 7: Handle Nulls (Interactive)
    # -------------------------------------------------------------
    print_step_header(7, "Handle Nulls (Interactive Selection)")
    df = handle_nulls_interactive(df, auto_mode=auto_mode)

    # -------------------------------------------------------------
    # Step 8: Check Outliers
    # -------------------------------------------------------------
    print_step_header(8, "Check Outliers")
    active_nums = [c for c in df.select_dtypes(include=["number"]).columns if c != target_col and df[c].nunique() > 10]
    check_outliers(df, num_cols=active_nums, target_col=target_col)

    # -------------------------------------------------------------
    # Step 9: Handle Outliers (Interactive)
    # -------------------------------------------------------------
    print_step_header(9, "Handle Outliers (Interactive Selection)")
    df = handle_outliers_interactive(df, num_cols=active_nums, target_col=target_col, auto_mode=auto_mode)

    # -------------------------------------------------------------
    # Step 10: Check Duplications
    # -------------------------------------------------------------
    print_step_header(10, "Check Duplications")
    check_duplicates(df)

    # -------------------------------------------------------------
    # Step 11: Handle Duplications
    # -------------------------------------------------------------
    print_step_header(11, "Handle Duplications")
    df = handle_duplicates(df, keep="first")

    # -------------------------------------------------------------
    # Step 12: Data Visualization (Plotly Dashboards & Distributions)
    # -------------------------------------------------------------
    print_step_header(12, "Data Visualization (Plotly Dashboards & Distributions)")
    dashboard_path = None
    if not skip_plots:
        active_cat_plot = [c for c in df.columns if c != target_col and (str(df[c].dtype) == "category" or df[c].dtype == object or df[c].nunique() <= 15)]
        active_num_plot = [c for c in df.select_dtypes(include=["number"]).columns if c != target_col and df[c].nunique() > 10]
        dashboard_path = generate_visualizations(
            df=df,
            output_dir=dataset_figures_dir,
            target_col=target_col,
            task_type=task_type,
            cat_cols=active_cat_plot,
            num_cols=active_num_plot,
            auto_open=auto_open
        )
    else:
        print("[Notice] Visualizations bypassed by user option.")

    # -------------------------------------------------------------
    # Step 13: Data Splitting
    # -------------------------------------------------------------
    print_step_header(13, "Data Splitting (X/y & Stratified/Random Split)")
    X, y = split_features_target(df, target_col=target_col)
    X_train, X_test, y_train, y_test = split_train_test_interactive(
        X=X,
        y=y,
        task_type=task_type,
        default_test_size=DEFAULT_TEST_SIZE,
        default_random_state=DEFAULT_RANDOM_STATE,
        auto_mode=auto_mode
    )

    # -------------------------------------------------------------
    # Step 14: Normalization / Feature Scaling (Interactive)
    # -------------------------------------------------------------
    print_step_header(14, "Normalization / Feature Scaling (Interactive Selection)")
    train_nums = X_train.select_dtypes(include=["number"]).columns.tolist()
    X_train, X_test, scaler = scale_numerical_features_interactive(
        X_train=X_train,
        X_test=X_test,
        num_cols=train_nums,
        auto_mode=auto_mode
    )

    # -------------------------------------------------------------
    # Step 15: Encoding (Categorical Encoding - Interactive)
    # -------------------------------------------------------------
    print_step_header(15, "Categorical Encoding (Interactive Selection)")
    train_cats = [
        c for c in X_train.columns
        if str(X_train[c].dtype) == "category" or X_train[c].dtype == object or X_train[c].nunique() <= 10
    ]
    X_train, X_test, encoder = encode_categorical_features_interactive(
        X_train=X_train,
        X_test=X_test,
        cat_cols=train_cats,
        auto_mode=auto_mode
    )

    # -------------------------------------------------------------
    # Export Clean Splits
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f" EXPORTING PREPROCESSED SPLITS FOR '{dataset_name.upper()}' ".center(70, "="))
    print("=" * 70)

    X_train_path = dataset_processed_dir / "X_train.csv"
    X_test_path = dataset_processed_dir / "X_test.csv"
    y_train_path = dataset_processed_dir / "y_train.csv"
    y_test_path = dataset_processed_dir / "y_test.csv"

    X_train.to_csv(X_train_path, index=False)
    X_test.to_csv(X_test_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    y_test.to_csv(y_test_path, index=False)

    print(f"[Exported] Training Features : {X_train_path.resolve()} (Shape: {X_train.shape})")
    print(f"[Exported] Testing Features  : {X_test_path.resolve()} (Shape: {X_test.shape})")
    print(f"[Exported] Training Targets   : {y_train_path.resolve()} (Shape: {y_train.shape})")
    print(f"[Exported] Testing Targets    : {y_test_path.resolve()} (Shape: {y_test.shape})")

    print("\n" + "#" * 70)
    print(" UNIVERSAL PIPELINE COMPLETED SUCCESSFULLY! ".center(70, "#"))
    print(f" Dataset Processed : {dataset_name}")
    print(f" Interactive Visuals: {dataset_figures_dir.resolve()}")
    print(f" Processed Datasets: {dataset_processed_dir.resolve()}")
    print(f" Ready for Model Training ({task_type.capitalize()})")
    print("#" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Universal 15-Step Tabular Machine Learning Preprocessing Framework."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to any CSV, TSV, or Excel dataset file."
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target column name to predict."
    )
    parser.add_argument(
        "--task",
        choices=["classification", "regression", "auto"],
        default="auto",
        help="Task type: 'classification', 'regression', or 'auto'."
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "auto"],
        default="interactive",
        help="Execution mode: 'interactive' prompts the user; 'auto' selects default methods."
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating Plotly HTML charts if fast execution is needed."
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Automatically open the unified visual dashboard in your browser upon completion."
    )
    args = parser.parse_args()

    auto_mode = (args.mode == "auto")
    run_pipeline(
        data_path=args.data,
        target_col=args.target,
        task_type=args.task,
        auto_mode=auto_mode,
        skip_plots=args.skip_plots,
        auto_open=args.view
    )


if __name__ == "__main__":
    import pandas as pd
    main()
