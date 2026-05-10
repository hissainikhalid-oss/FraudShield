"""
data_preprocessing.py
---------------------
Loads raw data, handles missing values, encodes
categorical columns, and saves the cleaned dataset.
"""

import pandas as pd
import numpy as np
import os


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV dataset."""
    print(f"[INFO] Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset shape: {df.shape}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with median (numeric) or mode (categorical)."""
    print("[INFO] Handling missing values...")
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ["float64", "int64"]:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    print(f"[INFO] Missing values after handling: {df.isnull().sum().sum()}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    print(f"[INFO] Removed {before - len(df)} duplicate rows.")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns."""
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        print(f"[INFO] Encoding categorical columns: {cat_cols}")
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df


def preprocess(raw_path: str, save_path: str) -> pd.DataFrame:
    """Full preprocessing pipeline."""
    df = load_data(raw_path)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = encode_categoricals(df)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[INFO] Processed data saved to: {save_path}")
    return df


if __name__ == "__main__":
    preprocess(
        raw_path="data/raw/creditcard.csv",
        save_path="data/processed/creditcard_cleaned.csv"
    )
