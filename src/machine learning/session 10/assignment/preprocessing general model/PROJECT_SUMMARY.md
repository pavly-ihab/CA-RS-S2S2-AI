# Universal Data Preprocessing Framework
## Comprehensive Project Summary & User Guide
*(Written for both Technical and Non-Technical Audiences)*

---

## 📌 Executive Summary (For Non-Technical Users & Stakeholders)

### What Is This Project?
In the world of Artificial Intelligence and Machine Learning, raw data collected from the real world is almost always messy, incomplete, full of typos, and improperly formatted. If you feed raw data directly into an AI model, the model will produce incorrect or biased predictions—a classic problem known in computing as **"Garbage In, Garbage Out"**.

This project is an **automated, intelligent data refinery**. It takes any raw tabular business dataset (e.g., customer churn, insurance claims, medical records, financial risk) and guides it through an **industry-standard 15-step sanitization process**. 

At the end of the pipeline, the dataset is transformed into a clean, mathematically optimized format ready for training state-of-the-art predictive AI models.

```
+------------------+       +-------------------------------+       +----------------------------+
|  Raw Messy Data  |  -->  | 15-Step Universal Auto-Engine |  -->  |  AI-Ready Training Splits  |
| (Missing, Noisy) |       |   (Interactive & Automated)   |       |  & Interactive Visuals     |
+------------------+       +-------------------------------+       +----------------------------+
```

### Why Is This Valuable?
1. **Saves Massive Time**: Data scientists typically spend 70–80% of their time manually writing repetitive data-cleaning code. This framework automates that entire process down to seconds.
2. **Interactive Human-in-the-Loop Control**: Unlike rigid black-box scripts, this framework acts as a digital advisor. It asks the user which method they prefer at critical decision gates (e.g., how to fill missing values or handle extreme anomalies).
3. **No Technical Expertise Required to Run**: Anyone on the business team can run the script with a single command and understand what is happening through plain-English terminal prompts.
4. **Visual Transparency**: It generates an **interactive visual dashboard** in your web browser, allowing stakeholders to visually explore feature distributions, anomalies, and correlations without writing a single line of code.

---

## 🔬 Technical Overview (For Data Scientists & Engineers)

### Architecture & Engineering Highlights
- **Universal Adaptability**: Agnostic to dataset size, column names, and problem types (supports both **Classification** and **Continuous Regression**).
- **Format Flexibility**: Ingests `.csv`, `.tsv`, `.txt`, `.xlsx`, and `.parquet` files natively.
- **Strict Data Leakage Prevention**: Data splitting (Step 13) occurs **before** feature scaling (Step 14) and categorical encoding (Step 15). All scalers and encoders are strictly `fit` exclusively on $X_{train}$ and only `transform` both $X_{train}$ and $X_{test}$.
- **Decoupled Modular Architecture**: Follows Single Responsibility and DRY principles. Every preprocessing phase lives in an isolated module inside `modules/`, making it unit-testable and extensible.
- **Interactive Terminal Wizard (`interactive_cli.py`)**: Provides input validation, graceful interrupt recovery, and an automated headless fallback mode (`--mode auto`).
- **Interactive Plotly Suite**: Replaces static image exports with responsive Plotly HTML charts compiled into a single master web application (`dashboard.html`).

---

## 🗺️ The 15 Preprocessing Steps: Explained for Everyone

| Step | Phase Name | What It Does (Non-Technical) | Technical Mechanics & Algorithms |
| :---: | :--- | :--- | :--- |
| **1** | **Business Understanding** | Defines what question we are trying to answer and how we measure success. | Sets ML problem type (Classification vs Regression), target variable $y$, and evaluation metrics (ROC-AUC/F1 vs RMSE/R²). |
| **2** | **Data Understanding** | Takes an initial inventory of the dataset (size, memory, preview). | Verifies data dimensions, memory footprint (`df.memory_usage(deep=True)`), and generates statistical descriptions. |
| **3** | **Drop Non-Interesting Features** | Removes useless data like IDs, serial numbers, or columns that are mostly blank. | Auto-identifies 100% unique keys or columns with >60% missing values; safely drops columns without KeyErrors. |
| **4** | **Check Data Types** | Examines how the computer is storing each column (numbers vs text). | Diagnostic profiling of pandas dtypes, cardinality counts, and non-null sample inspection. |
| **5** | **Handle Data Types** | Ensures numbers representing categories (e.g. 1st/2nd/3rd class) are treated as categories, not math. | Casts discrete variables into pandas `'category'` dtype for memory optimization and downstream encoder mapping. |
| **6** | **Check Nulls** | Counts how many blank/missing values exist in each column. | Tabulates absolute missingness and percentage ratios (`df.isnull().sum() / len(df)`). |
| **7** | **Handle Nulls (Interactive)** | Fills in missing values using smart strategies (averages, most common, or dropping). | Interactive choice: Grouped Median (by sub-features), Global Median, Mean, Mode, or Constant placeholder. |
| **8** | **Check Outliers** | Finds weird, extreme numbers (e.g., a passenger who paid $500 when normal is $20). | Dual diagnostic: Interquartile Range fences ($Q1 - 1.5 \times IQR$, $Q3 + 1.5 \times IQR$) and Z-score tests ($|Z| > 3.0$). |
| **9** | **Handle Outliers (Interactive)** | Softens or removes extreme values so they don't distort the AI model. | Interactive choice: **Winsorization/Clipping** (caps values to fences), **Median replacement**, **Mean replacement**, or **Trimming**. |
| **10** | **Check Duplications** | Scans for identical repeated records that might bias results. | Row-level duplicate inspection using `df.duplicated().sum()`. |
| **11** | **Handle Duplications** | Safely removes duplicate rows, keeping only one clean record. | Deduplicates records using `drop_duplicates(keep='first')`. |
| **12** | **Data Visualization** | Generates an interactive visual website showing graphs of every feature. | Plotly engine compiling donut charts, multi-panel grids, outlier boxplots, histograms, and heatmaps into `dashboard.html`. |
| **13** | **Data Splitting** | Splits the data into two sets: one for study (Training) and one for testing (Exam). | Separates $X$ and $y$. Performs **Stratified Split** for classification (preserves class ratios) or random split for regression. |
| **14** | **Normalization / Scaling** | Brings all numbers onto the same playing field (e.g. 0 to 1) so large numbers don't overpower small ones. | User choice: `MinMaxScaler` $[0, 1]$, `StandardScaler` ($\mu=0, \sigma=1$), or `RobustScaler` (median & IQR). Fit on train, transform on test. |
| **15** | **Categorical Encoding** | Translates text categories (like 'Male'/'Female' or 'North'/'South') into mathematical numbers the AI can read. | User choice: `OneHotEncoder` (binary columns), Dummy encoding (`drop_first=True`), or `OrdinalEncoder`. Fit on train, transform on test. |

---

## 📊 The Master Visual Dashboard (`dashboard.html`)

Instead of forcing users to hunt down and open dozens of individual files, the framework compiles all visualizations into a **single, unified, responsive web dashboard**:

1. **Target Distribution Card**: Donut chart for classification balance or continuous histogram for regression.
2. **Categorical Features Overview**: Multi-panel grid (`make_subplots`) summarizing all discrete categories.
3. **Outlier Box Plots**: Interactive box plots displaying median lines, quartiles, and individual jittered data points (`points='all'`).
4. **Continuous Distributions**: Histograms layered with marginal boxplots showing spread and skewness.
5. **Correlation Heatmap**: Color-coded matrix showing how features relate to one another (identifies multicollinearity).

---

## 💻 User Execution Guide

### 1. Interactive Mode (Recommended for Human-in-the-loop)
Run the script from your terminal:
```powershell
python main.py
```
- **What happens**: The system asks for your dataset path (press `Enter` to use the built-in Titanic data).
- At each step, it displays a numbered menu asking how you would like to handle missing values, outliers, scaling, and encoding.

### 2. Run on Any New Dataset
```powershell
python main.py --data "path/to/my_data.csv" --target "TargetColumn"
```

### 3. Fully Automated Mode (Zero Prompts)
If you want to run the pipeline inside an automated script or background job:
```powershell
python main.py --mode auto
```

### 4. Auto-Open Visual Dashboard in Browser
Add the `--view` flag to have the dashboard automatically pop open in Chrome or Edge upon completion:
```powershell
python main.py --view
```

---

## 📂 Output Artifacts Created by the System

Every run automatically generates cleanly organized files under `data/processed/` and `reports/figures/`:

```
reports/figures/<dataset_name>/
└── dashboard.html               <-- Complete interactive visual dashboard in 1 file!

data/processed/<dataset_name>/
├── X_train.csv                  <-- Clean, scaled, encoded training features
├── X_test.csv                   <-- Clean, scaled, encoded testing features
├── y_train.csv                  <-- Training target labels
└── y_test.csv                   <-- Testing target labels
```
These four CSV files are **100% ready** to be fed into any machine learning algorithm (`LogisticRegression`, `LinearRegression`, `RandomForest`, `XGBoost`, `LightGBM`, or Neural Networks) without needing any additional cleaning!
