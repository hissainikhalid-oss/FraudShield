"""
evaluate.py
-----------
Loads the saved model and evaluates it on the test set.
Prints full classification report, confusion matrix,
and plots ROC curve + feature importances.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score,
    accuracy_score, precision_score, recall_score
)


def load_model_and_data(model_path: str, features_path: str):
    """Load saved model and test data."""
    model = joblib.load(model_path)
    df = pd.read_csv(features_path)
    X = df.drop("Class", axis=1)
    y = df["Class"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)
    print(f"[INFO] Model loaded: {type(model).__name__}")
    return model, X_test, y_test


def print_metrics(y_test, y_pred, y_prob):
    """Print all evaluation metrics."""
    print("\n" + "="*50)
    print(" EVALUATION RESULTS")
    print("="*50)
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"  F1-Score  : {f1_score(y_test, y_pred):.4f}")
    print(f"  AUC-ROC   : {roc_auc_score(y_test, y_prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
          target_names=["Legitimate", "Fraud"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def plot_roc_curve(y_test, y_prob, save_path: str = "models/roc_curve.png"):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="#1F4E79", lw=2, label=f"AUC = {auc:.3f}")
    plt.plot([0,1],[0,1],"k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Fraud Detection Model")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] ROC curve saved to: {save_path}")


def plot_feature_importance(model, feature_names,
                            save_path: str = "models/feature_importance.png"):
    """Plot top 10 feature importances (Random Forest)."""
    if not hasattr(model, "feature_importances_"):
        print("[INFO] Model does not support feature importances. Skipping.")
        return
    importances = pd.Series(
        model.feature_importances_, index=feature_names
    ).sort_values(ascending=False)[:10]

    plt.figure(figsize=(8, 5))
    importances.plot(kind="barh", color="#1F4E79", edgecolor="white")
    plt.gca().invert_yaxis()
    plt.title("Top 10 Feature Importances")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Feature importance saved to: {save_path}")


def evaluate(model_path: str = "models/fraud_model.pkl",
             features_path: str = "data/processed/creditcard_features.csv"):
    """Full evaluation pipeline."""
    model, X_test, y_test = load_model_and_data(model_path, features_path)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print_metrics(y_test, y_pred, y_prob)
    plot_roc_curve(y_test, y_prob)
    plot_feature_importance(model, X_test.columns.tolist())


if __name__ == "__main__":
    evaluate()
