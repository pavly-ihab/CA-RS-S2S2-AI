# main.py
from config.config import DATA_PATH, COLS_TO_DROP, CAT_COLS
from preprocessing import (
    read_data_file,
    drop_unnecessary_features,
    check_data_type,
    convert_to_categorical
)

def main():
    print("=" * 60)
    print("STEP 1: Reading Dataset")
    print("=" * 60)
    df = read_data_file(DATA_PATH)

    if df is None:
        print("[Pipeline Aborted] Could not proceed without valid data.")
        return

    print("\n" + "=" * 60)
    print("STEP 2: Initial Data Inspection")
    print("=" * 60)
    initial_report = check_data_type(df)
    print(initial_report)

    print("\n" + "=" * 60)
    print("STEP 3: Dropping Unnecessary Features")
    print("=" * 60)
    df = drop_unnecessary_features(df, cols_to_drop=COLS_TO_DROP)

    print("\n" + "=" * 60)
    print("STEP 4: Converting Target Columns to Categorical")
    print("=" * 60)
    df = convert_to_categorical(df, cat_cols=CAT_COLS)

    print("\n" + "=" * 60)
    print("STEP 5: Final Data Quality Report")
    print("=" * 60)
    final_report = check_data_type(df)
    print(final_report)

if __name__ == "__main__":
    main()