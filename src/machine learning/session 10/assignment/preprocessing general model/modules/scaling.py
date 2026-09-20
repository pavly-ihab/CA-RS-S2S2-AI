# modules/scaling.py
"""
Step 14: Universal Feature Scaling & Normalization (Interactive)
Prompts user for numerical scaling method and fits strictly on X_train,
transforming both train and test splits to eliminate data leakage across any dataset.
"""
from typing import List, Tuple, Optional, Any
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from modules.interactive_cli import prompt_user_choice


def scale_numerical_features_interactive(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    num_cols: Optional[List[str]] = None,
    auto_mode: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[Any]]:
    """
    Step 14: Dynamically scales numerical features for any dataset.
    """
    if num_cols is None:
        # Detect numeric columns in X_train
        num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()

    valid_cols = [c for c in num_cols if c in X_train.columns and pd.api.types.is_numeric_dtype(X_train[c])]

    if not valid_cols:
        print("\n--- Feature Normalization / Scaling ---")
        print("[Notice] No numerical columns available to scale.")
        return X_train, X_test, None

    print("\n--- Feature Normalization / Scaling Configuration ---")
    options = {
        "1": "MinMaxScaler (Binds continuous features into [0, 1])",
        "2": "StandardScaler (Standardizes to mean=0, std=1)",
        "3": "RobustScaler (Scales with median & IQR; robust to outliers)",
        "4": "None / Skip Scaling"
    }

    choice = prompt_user_choice(
        prompt_text=f"Select scaling method for {len(valid_cols)} numeric features {valid_cols}:",
        options=options,
        default_key="1",
        auto_mode=auto_mode
    )

    if choice == "4":
        print("[Notice] Feature scaling skipped.")
        return X_train, X_test, None

    scaler_map = {
        "1": ("MinMaxScaler", MinMaxScaler()),
        "2": ("StandardScaler", StandardScaler()),
        "3": ("RobustScaler", RobustScaler())
    }

    scaler_name, scaler = scaler_map[choice]

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    scaler.fit(X_train[valid_cols])
    X_train_scaled[valid_cols] = scaler.transform(X_train[valid_cols])
    X_test_scaled[valid_cols] = scaler.transform(X_test[valid_cols])

    print(f"\n[Success] Applied {scaler_name} on {valid_cols}:")
    print("Train Sample After Scaling:")
    print(X_train_scaled[valid_cols].head(3))

    return X_train_scaled, X_test_scaled, scaler
