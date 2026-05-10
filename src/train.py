"""
train.py
--------
Trains Logistic Regression, Decision Tree, and
Random Forest models on the processed dataset.
Saves the best model (Random Forest) to models/.
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
def load_features(path: str):
    """Load processed feature dataset."""
    df = pd.read_csv(path)
    X = df.drop("Class", axis=1)
    y = df["Class"]
    print(f"[INFO] Features shape: {X.shape}")
    print(f"[INFO] Class distribution:\n{y.value_counts()}")
    return X, y
def get_models() -> dict:
    """Return dictionary of models to train."""
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=500, random_state=42),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", n_estimators=100,
            random_state=42, n_jobs=-1),
    }
def train_and_evaluate(X_train, X_test, y_train, y_test) -> dict:
    """Train all models and return results."""
    models = get_models()
    results = {}
    for name, model in models.items():
        print(f"\n[INFO] Training: {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "model":  model,
            "f1":     round(f1_score(y_test, y_pred), 4),
            "auc":    round(roc_auc_score(y_test, y_prob), 4),
        }
        print(f"  F1-Score : {results[name]['f1']}")
        print(f"  AUC-ROC  : {results[name]['auc']}")
    return results
def save_best_model(results: dict,
                    model_path: str = "models/fraud_model.pkl"):
    """Save the model with highest F1-Score."""
    best_name = max(results, key=lambda k: results[k]["f1"])
    best_model = results[best_name]["model"]
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
    print(f"\n[INFO] Best model: {best_name}")
    print(f"[INFO] Saved to: {model_path}")
    return best_model
def train(features_path: str = "data/processed/creditcard_features.csv",
          model_path: str = "models/fraud_model.pkl"):
    """Full training pipeline."""
    X, y = load_features(features_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)
    print(f"[INFO] Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    results = train_and_evaluate(X_train, X_test, y_train, y_test)
    best_model = save_best_model(results, model_path)
    return best_model, X_test, y_test
if __name__ == "__main__":
    train()
