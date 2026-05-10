"""
main.py
-------
Master pipeline script. Runs all steps in order:
  1. Data Preprocessing
  2. Feature Engineering
  3. Model Training
  4. Model Evaluation

Run: python main.py
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.data_preprocessing  import preprocess
from src.feature_engineering import engineer_features
from src.train               import train
from src.evaluate            import evaluate


def run_pipeline():
    print("\n" + "="*55)
    print("   FINANCIAL FRAUD DETECTION — ML PIPELINE")
    print("="*55)

    # ── STEP 1: Preprocessing ──────────────────────────
    print("\n▶ STEP 1: Data Preprocessing")
    preprocess(
        raw_path  ="data/raw/creditcard.csv",
        save_path ="data/processed/creditcard_cleaned.csv"
    )

    # ── STEP 2: Feature Engineering ───────────────────
    print("\n▶ STEP 2: Feature Engineering")
    engineer_features(
        cleaned_path="data/processed/creditcard_cleaned.csv",
        save_path   ="data/processed/creditcard_features.csv"
    )

    # ── STEP 3: Training ──────────────────────────────
    print("\n▶ STEP 3: Model Training")
    train(
        features_path="data/processed/creditcard_features.csv",
        model_path   ="models/fraud_model.pkl"
    )

    # ── STEP 4: Evaluation ────────────────────────────
    print("\n▶ STEP 4: Model Evaluation")
    evaluate(
        model_path   ="models/fraud_model.pkl",
        features_path="data/processed/creditcard_features.csv"
    )

    print("\n" + "="*55)
    print("   PIPELINE COMPLETE ✓")
    print("="*55)


if __name__ == "__main__":
    run_pipeline()
