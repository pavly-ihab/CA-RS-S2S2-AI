# Universal Tabular Machine Learning Preprocessing Framework

A production-grade, modular Python framework executing all **15 Industry-Standard Preprocessing Steps** sequentially for **ANY tabular dataset** (Classification & Regression).

Built with interactive terminal decision gates, automated profiling, data-leakage protection, and comprehensive interactive Plotly visual dashboards based on **Sessions 8, 9, and 10**.

---

## 📁 Project Architecture

```
project_titanic_advanced/
├── config/
│   ├── __init__.py
│   └── config.py               # Dynamic paths & universal defaults
├── data/
│   ├── raw/
│   │   └── Titanic.csv         # Raw fallback dataset
│   └── processed/              # Exported clean splits organized by dataset:
│       ├── Titanic/            # (X_train.csv, X_test.csv, y_train.csv, y_test.csv)
│       └── Salary_Data/        # (X_train.csv, X_test.csv, y_train.csv, y_test.csv)
├── reports/
│   └── figures/                # Interactive Plotly HTML dashboards organized by dataset:
│       ├── Titanic/            # 7 interactive HTML charts
│       └── Salary_Data/        # 4 interactive HTML charts
├── modules/
│   ├── __init__.py
│   ├── interactive_cli.py      # Terminal menu wizard, column pickers, and decision gates
│   ├── business_understanding.py  # Step 1 (Classification or Regression framing)
│   ├── data_understanding.py      # Step 2 (Universal format loader & column profiler)
│   ├── feature_selection.py       # Step 3 (Interactive feature dropping)
│   ├── dtype_management.py        # Steps 4 & 5 (Universal dtype check & categorical casting)
│   ├── missing_values.py          # Steps 6 & 7 (Universal interactive imputation)
│   ├── outliers.py                # Steps 8 & 9 (Universal interactive outlier treatment)
│   ├── duplicates.py              # Steps 10 & 11 (Universal row deduplication)
│   ├── visualizer.py              # Step 12 (Universal Plotly Dashboards & Distributions)
│   ├── data_splitting.py          # Step 13 (Stratified for Classification, Random for Regression)
│   ├── scaling.py                 # Step 14 (Interactive Feature Scaling)
│   └── encoding.py                # Step 15 (Interactive Categorical Encoding)
├── main.py                     # Master universal pipeline entrypoint
├── README.md                   # Full documentation
└── requirements.txt            # Package dependencies
```

---

## 🚀 How to Run with ANY Dataset

### 1. Interactive Launch (Prompts you for dataset path, target, and methods)
```bash
python main.py
```
*If you simply press Enter, it defaults to the bundled Titanic dataset. You can also paste the path to any CSV, TSV, or Excel file on your machine.*

### 2. Specify Any Dataset via CLI
```bash
# Example: Run on another dataset (e.g. Salary_Data.csv)
python main.py --data "path/to/any_dataset.csv" --target "TargetColumnName"
```

### 3. Fully Automated Mode (Auto-detects target, task, and optimal methods)
```bash
# Automated run on Titanic (Classification)
python main.py --mode auto

# Automated run on a Regression dataset
python main.py --data "path/to/Salary_Data.csv" --target "Salary" --task regression --mode auto
```

### 4. Fast Mode (Skip Plotly HTML chart generation)
```bash
python main.py --data "path/to/dataset.csv" --skip-plots
```

---

## 📊 Preprocessing Steps Covered for Any Dataset

1. **Business Understanding**: Dynamic problem framing (Supervised Classification vs. Regression) and metric selection.
2. **Data Understanding**: Multi-format ingestion (`.csv`, `.tsv`, `.xlsx`, `.parquet`), memory analysis, descriptive statistics, and automated feature profiling (ID candidates, high-sparsity columns, categorical vs. numerical features).
3. **Drop Non-Interesting Features**: Interactive column picker with auto-suggested candidate ID/sparse columns.
4. **Check dtypes**: Diagnostic report of data types, cardinalities, and sample values.
5. **Handle dtypes**: Interactive casting of discrete columns to pandas `'category'`.
6. **Check Nulls**: Tabulation of missing values counts and percentages.
7. **Handle Nulls (Interactive)**:
   - **Numerical**: Median, Mean, Mode, Constant (0), or Row Dropping.
   - **Categorical**: Mode (most frequent), Unknown category placeholder, or Row Dropping.
8. **Check Outliers**: IQR fence computation ($Q1 - 1.5 \times IQR$, $Q3 + 1.5 \times IQR$) and Z-score diagnostic ($|Z| > 3$) for all continuous numerical features.
9. **Handle Outliers (Interactive)**:
   - Clipping/Winsorizing to IQR fences.
   - Replacing with median (Session 9 standard).
   - Replacing with mean.
   - Dropping outlier rows.
   - Keeping raw values.
10. **Check Duplications**: Row-level duplication inspection.
11. **Handle Duplications**: Deduplication keeping first occurrences.
12. **Data Visualization (Plotly)**:
    - Target distribution: Donut chart for classification; Histogram + Marginal Box for regression.
    - Categorical subplots: Multi-panel grid layout (`make_subplots`) for all discrete features.
    - Outlier box plots: Jittered data points (`points='all'`) for each continuous variable.
    - Continuous feature distributions: Histograms with marginal box plots.
    - Correlation heatmap: Pearson correlation matrix for all numeric features.
13. **Data Splitting**: Stratified split for classification (preserving class proportions) or random split for regression, preventing data leakage.
14. **Normalization / Scaling (Interactive)**:
    - User choice of `MinMaxScaler`, `StandardScaler`, `RobustScaler`, or None. Fitted strictly on $X_{train}$ and transformed on $X_{test}$.
15. **Encoding (Interactive)**:
    - User choice of `OneHotEncoder` (via `category_encoders` / `sklearn`), Dummy encoding (`drop_first`), `OrdinalEncoder`, or None. Fitted on $X_{train}$ and transformed on $X_{test}$.
