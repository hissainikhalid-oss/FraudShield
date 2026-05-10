"""
feature_engineering.py
-----------------------
Creates new features from existing columns to improve
model performance on fraud detection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour of day from Time column."""
    if "Time" in df.columns:
        df["Hour"] = (df["Time"] // 3600).astype(int) % 24
        df["Is_Night"] = df["Hour"].apply(lambda h: 1 if h < 6 or h >= 22 else 0)
        print("[INFO] Time features added: Hour, Is_Night")
    return df


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Log-transform Amount to reduce skewness."""
    if "Amount" in df.columns:
        df["Amount_Log"] = np.log1p(df["Amount"])
        df["Amount_High"] = (df["Amount"] > df["Amount"].quantile(0.95)).astype(int)
        print("[INFO] Amount features added: Amount_Log, Amount_High")
    return df


def scale_features(df: pd.DataFrame,
                   target_col: str = "Class",
                   scaler_path: str = "models/scaler.pkl") -> pd.DataFrame:
    """StandardScale all numeric features except the target."""
    feature_cols = [c for c in df.columns if c != target_col]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"[INFO] Scaler saved to: {scaler_path}")
    return df


def engineer_features(cleaned_path: str,
                       save_path: str) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df = pd.read_csv(cleaned_path)
    df = add_time_features(df)
    df = add_amount_features(df)
    df = scale_features(df)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[INFO] Engineered features saved to: {save_path}")
    return df


if __name__ == "__main__":
    engineer_features(
        cleaned_path="data/processed/creditcard_cleaned.csv",
        save_path="data/processed/creditcard_features.csv"
    )
