import pandas as pd


def read_files(file_path):
    return pd.read_csv(file_path)


def drop_cols(df, cols):
    return df.drop(columns=cols)


def get_type_info(df):
    return pd.DataFrame({"dtypes": df.dtypes, "nunique": df.nunique()}).T
