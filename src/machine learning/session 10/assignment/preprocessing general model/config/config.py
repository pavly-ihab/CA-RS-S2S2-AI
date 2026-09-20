# config/config.py
"""
Universal Configuration Module
Defines dynamic paths, pipeline defaults, and dataset-agnostic configurations.
"""
from pathlib import Path
from typing import Optional

# Base Directory Resolution
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_RAW_DATA_PATH = DATA_DIR / "raw" / "Titanic.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure output directories exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Default Pipeline Settings
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_IQR_MULTIPLIER = 1.5
DEFAULT_Z_THRESHOLD = 3.0
MAX_CATEGORICAL_CARDINALITY = 25  # Features with fewer unique values can be treated as categorical
HIGH_NULL_THRESHOLD_PCT = 60.0    # Features with missing values exceeding this % are flagged for review
