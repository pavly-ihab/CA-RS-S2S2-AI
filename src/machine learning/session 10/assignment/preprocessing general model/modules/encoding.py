# modules/encoding.py
"""
Step 15: Universal Categorical Encoding (Interactive)
Dynamically identifies discrete/string features across any dataset,
prompts user for encoding method (One-Hot, Dummy drop_first, Ordinal),
and fits on X_train while transforming X_test to guarantee identical schema.
"""
from typing import List, Tuple, Optional, Any
import pandas as pd
from modules.interactive_cli import prompt_user_choice


def encode_categorical_features_interactive(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_cols: Optional[List[str]] = None,
    auto_mode: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[Any]]:
    """
    Step 15: Dynamically encodes categorical features for any dataset.
    """
    if cat_cols is None:
        cat_cols = [
            c for c in X_train.columns
            if (str(X_train[c].dtype) == "category" or
                X_train[c].dtype == object or
                (pd.api.types.is_numeric_dtype(X_train[c]) and X_train[c].nunique() <= 10))
        ]

    target_cats = [c for c in cat_cols if c in X_train.columns]

    if not target_cats:
        print("\n--- Categorical Encoding ---")
        print("[Notice] No categorical features found to encode.")
        return X_train, X_test, None

    print("\n--- Categorical Encoding Configuration ---")
    options = {
        "1": "One-Hot Encoding (Full representation - Session 10 standard)",
        "2": "One-Hot / Dummy Encoding (drop_first=True - avoids collinearity)",
        "3": "Ordinal / Integer Encoding (Compact label ordering)",
        "4": "None / Skip Encoding"
    }

    choice = prompt_user_choice(
        prompt_text=f"Select encoding strategy for {len(target_cats)} features {target_cats}:",
        options=options,
        default_key="1",
        auto_mode=auto_mode
    )

    if choice == "4":
        print("[Notice] Encoding skipped.")
        return X_train, X_test, None

    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()

    try:
        import category_encoders as ce
        has_ce = True
    except ImportError:
        has_ce = False

    if choice == "1":
        if has_ce:
            encoder = ce.OneHotEncoder(cols=target_cats, use_cat_names=True, drop_invariant=True)
            X_train_enc = encoder.fit_transform(X_train_enc)
            X_test_enc = encoder.transform(X_test_enc)
            encoder_name = "category_encoders.OneHotEncoder"
        else:
            from sklearn.preprocessing import OneHotEncoder
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            train_enc = encoder.fit_transform(X_train_enc[target_cats])
            test_enc = encoder.transform(X_test_enc[target_cats])
            cols = encoder.get_feature_names_out(target_cats)

            train_df = pd.DataFrame(train_enc, columns=cols, index=X_train_enc.index)
            test_df = pd.DataFrame(test_enc, columns=cols, index=X_test_enc.index)

            X_train_enc = pd.concat([X_train_enc.drop(columns=target_cats), train_df], axis=1)
            X_test_enc = pd.concat([X_test_enc.drop(columns=target_cats), test_df], axis=1)
            encoder_name = "sklearn.preprocessing.OneHotEncoder"

    elif choice == "2":
        from sklearn.preprocessing import OneHotEncoder
        encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        train_enc = encoder.fit_transform(X_train_enc[target_cats])
        test_enc = encoder.transform(X_test_enc[target_cats])
        cols = encoder.get_feature_names_out(target_cats)

        train_df = pd.DataFrame(train_enc, columns=cols, index=X_train_enc.index)
        test_df = pd.DataFrame(test_enc, columns=cols, index=X_test_enc.index)

        X_train_enc = pd.concat([X_train_enc.drop(columns=target_cats), train_df], axis=1)
        X_test_enc = pd.concat([X_test_enc.drop(columns=target_cats), test_df], axis=1)
        encoder_name = "OneHotEncoder (drop='first')"

    elif choice == "3":
        from sklearn.preprocessing import OrdinalEncoder
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train_enc[target_cats] = encoder.fit_transform(X_train_enc[target_cats])
        X_test_enc[target_cats] = encoder.transform(X_test_enc[target_cats])
        encoder_name = "sklearn.preprocessing.OrdinalEncoder"

    print(f"\n[Success] Applied {encoder_name}:")
    print(f"  - Features before: {X_train.shape[1]} -> Features after: {X_train_enc.shape[1]}")
    print(f"  - Encoded Feature List: {list(X_train_enc.columns)}")

    return X_train_enc, X_test_enc, encoder
