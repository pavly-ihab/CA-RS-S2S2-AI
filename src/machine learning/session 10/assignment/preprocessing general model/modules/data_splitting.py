# modules/data_splitting.py
"""
Step 13: Universal Data Splitting
Separates feature matrix X and target y, and executes stratified (for classification)
or standard random split (for regression) to strictly eliminate data leakage.
"""
from typing import Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from modules.interactive_cli import prompt_float, prompt_user_choice


def split_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separates the input DataFrame into X (features) and y (target).
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame columns: {df.columns.tolist()}")

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    return X, y


def split_train_test_interactive(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str = "classification",
    default_test_size: float = 0.2,
    default_random_state: int = 42,
    auto_mode: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Step 13: Interactively configures and performs train/test split.
    """
    print("\n--- Train / Test Splitting Configuration ---")
    test_size = prompt_float(
        prompt_text="Enter test set ratio (e.g., 0.2 for 80/20 split)",
        default_val=default_test_size,
        min_val=0.05,
        max_val=0.5,
        auto_mode=auto_mode
    )

    is_classification = (task_type.lower() == "classification")
    strat = None

    if is_classification:
        # Check if least frequent class has at least 2 instances for stratification
        min_class_count = y.value_counts().min()
        if min_class_count >= 2:
            stratify_choice = prompt_user_choice(
                prompt_text="Use stratified sampling to preserve target class proportions?",
                options={
                    "1": "Yes (Recommended for classification)",
                    "2": "No (Simple random split)"
                },
                default_key="1",
                auto_mode=auto_mode
            )
            strat = y if stratify_choice == "1" else None
        else:
            print("[Notice] Rare classes with < 2 instances detected. Using unstratified split.")
            strat = None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=default_random_state,
        stratify=strat
    )

    print(f"\n[Success] Dataset split successfully:")
    print(f"  - X_train Shape: {X_train.shape} | y_train: {y_train.shape}")
    print(f"  - X_test  Shape: {X_test.shape}  | y_test:  {y_test.shape}")

    if is_classification:
        train_rates = (y_train.value_counts(normalize=True) * 100).round(2).to_dict()
        test_rates = (y_test.value_counts(normalize=True) * 100).round(2).to_dict()
        print(f"  - Train Class Balance %: {train_rates}")
        print(f"  - Test Class Balance % : {test_rates}")

    return X_train, X_test, y_train, y_test
