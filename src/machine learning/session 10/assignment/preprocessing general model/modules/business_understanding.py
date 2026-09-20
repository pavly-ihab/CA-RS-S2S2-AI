# modules/business_understanding.py
"""
Step 1: Universal Business Understanding
Dynamically frames the ML objective, target variable, task type (Classification vs Regression),
and standard evaluation metrics for any dataset.
"""
from typing import Dict, Any, Optional
import pandas as pd


def print_business_understanding(
    target_col: str,
    task_type: str,
    dataset_name: str,
    df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Displays and returns dynamic business problem framing for any dataset.
    """
    is_classification = (task_type.lower() == "classification")

    if is_classification:
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        prob_desc = "Supervised Machine Learning - Classification"
        if df is not None and target_col in df.columns:
            n_classes = df[target_col].nunique()
            class_counts = df[target_col].value_counts(normalize=True) * 100
            balance_note = f"{n_classes} classes. Distribution: " + ", ".join([f"{k}: {v:.1f}%" for k, v in class_counts.items()])
        else:
            balance_note = "Classification target distribution will be evaluated."
    else:
        metrics = ["Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)", "R² Score"]
        prob_desc = "Supervised Machine Learning - Regression"
        balance_note = "Continuous numerical target. Minimizing prediction residuals."

    spec = {
        "dataset_name": dataset_name,
        "target_variable": target_col,
        "problem_type": prob_desc,
        "objective": f"Predict target '{target_col}' using preprocessed input features.",
        "primary_metrics": metrics,
        "target_profile": balance_note
    }

    print(f"Problem Formulation for '{dataset_name}':")
    print(f"  - Target Variable : {spec['target_variable']}")
    print(f"  - Task Type       : {spec['problem_type']}")
    print(f"  - Objective       : {spec['objective']}")
    print(f"  - Target Profile  : {spec['target_profile']}")
    print(f"  - Success Metrics : {', '.join(spec['primary_metrics'])}")

    return spec
