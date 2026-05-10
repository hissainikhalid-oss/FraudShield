import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# OPTIONAL ADVANCED MODELS
# pip install xgboost lightgbm catboost

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# ════════════════════════════════════════════════════
# LOAD DATASET
# ════════════════════════════════════════════════════

print("\n" + "="*60)
print("LOADING DATASET")
print("="*60)

data = pd.read_csv("data/raw/creditcard.csv")

print("Dataset Loaded Successfully!")
print("Dataset Shape:", data.shape)

print("\nFirst 5 Rows:")
print(data.head())

print("\nMissing Values:")
print(data.isnull().sum())

print("\nClass Distribution:")
print(data["Class"].value_counts())


# ════════════════════════════════════════════════════
# DATA VISUALIZATION
# ════════════════════════════════════════════════════

print("\n" + "="*60)
print("DATA VISUALIZATION")
print("="*60)

plt.figure(figsize=(6,4))

sns.countplot(x="Class", data=data)

plt.title("Fraud vs Legitimate Transactions")

plt.xticks([0,1], ["Legitimate", "Fraud"])

plt.show()


plt.figure(figsize=(10,5))

sns.histplot(data["Amount"], bins=50)

plt.title("Transaction Amount Distribution")

plt.xlabel("Amount")

plt.show()


# ════════════════════════════════════════════════════
# FEATURES AND TARGET
# ════════════════════════════════════════════════════

X = data.drop("Class", axis=1)
y = data["Class"]


# ════════════════════════════════════════════════════
# TRAIN TEST SPLIT
# ════════════════════════════════════════════════════

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain Size:", X_train.shape[0])
print("Test Size :", X_test.shape[0])


# ════════════════════════════════════════════════════
# MULTIPLE MODELS
# ════════════════════════════════════════════════════

print("\n" + "="*60)
print("TRAINING MULTIPLE MODELS")
print("="*60)

models = {

    "Logistic Regression":
        LogisticRegression(
            class_weight='balanced',
            max_iter=500,
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            class_weight='balanced',
            max_depth=6,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(),

    "AdaBoost":
        AdaBoostClassifier(),

    "SVM":
        SVC(
            probability=True
        ),

    "KNN":
        KNeighborsClassifier(),

    "XGBoost":
        XGBClassifier(
            eval_metric='logloss'
        ),

    "LightGBM":
        LGBMClassifier(),

    "CatBoost":
        CatBoostClassifier(
            verbose=0
        )
}

results = []

best_model_name = None
best_model = None
best_f1 = 0


# ════════════════════════════════════════════════════
# TRAIN & EVALUATE ALL MODELS
# ════════════════════════════════════════════════════

for name, model in models.items():

    print("\n" + "-"*60)
    print(f"MODEL: {name}")
    print("-"*60)

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    prob = model.predict_proba(X_test)[:,1]

    accuracy = accuracy_score(y_test, pred)

    precision = precision_score(
        y_test,
        pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        pred,
        zero_division=0
    )

    auc = roc_auc_score(y_test, prob)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"AUC-ROC  : {auc:.4f}")

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "AUC-ROC": auc
    })

    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_model = model


# ════════════════════════════════════════════════════
# MODEL COMPARISON TABLE
# ════════════════════════════════════════════════════

print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

results_df = pd.DataFrame(results)

print(results_df)


# ════════════════════════════════════════════════════
# BEST MODEL
# ════════════════════════════════════════════════════

print("\n" + "="*60)
print("BEST MODEL")
print("="*60)

print("Best Model:", best_model_name)

print("Best F1-Score:", round(best_f1, 4))


# ════════════════════════════════════════════════════
# HYPERPARAMETER TUNING
# ONLY FOR RANDOM FOREST
# ════════════════════════════════════════════════════

print("\n" + "="*60)
print("HYPERPARAMETER TUNING")
print("="*60)

param_grid = {

    'n_estimators': [100, 200],

    'max_depth': [10, 20],

    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(

    estimator=RandomForestClassifier(
        class_weight='balanced',
        random_state=42
    ),

    param_grid=param_grid,

    cv=3,

    scoring='f1',

    n_jobs=-1,

    verbose=1
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross Validation F1:")
print(round(grid_search.best_score_, 4))


# ════════════════════════════════════════════════════
# FINAL MODEL
# ════════════════════════════════════════════════════

print("\n" + "="*60)
print("FINAL MODEL")
print("="*60)

best_params = grid_search.best_params_

final_model = RandomForestClassifier(

    n_estimators=best_params['n_estimators'],

    max_depth=best_params['max_depth'],

    min_samples_split=best_params['min_samples_split'],

    class_weight='balanced',

    random_state=42
)

final_model.fit(X_train, y_train)

final_pred = final_model.predict(X_test)

final_prob = final_model.predict_proba(X_test)[:,1]


# ════════════════════════════════════════════════════
# FINAL METRICS
# ════════════════════════════════════════════════════

final_accuracy = accuracy_score(y_test, final_pred)

final_precision = precision_score(
    y_test,
    final_pred,
    zero_division=0
)

final_recall = recall_score(
    y_test,
    final_pred,
    zero_division=0
)

final_f1 = f1_score(
    y_test,
    final_pred,
    zero_division=0
)

final_auc = roc_auc_score(y_test, final_prob)

print(f"Accuracy : {final_accuracy:.4f}")

print(f"Precision: {final_precision:.4f}")

print(f"Recall   : {final_recall:.4f}")

print(f"F1-Score : {final_f1:.4f}")

print(f"AUC-ROC  : {final_auc:.4f}")


# ════════════════════════════════════════════════════
# CLASSIFICATION REPORT
# ════════════════════════════════════════════════════

print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)

print(classification_report(
    y_test,
    final_pred,
    target_names=['Legitimate', 'Fraud'],
    zero_division=0
))


# ════════════════════════════════════════════════════
# CONFUSION MATRIX
# ════════════════════════════════════════════════════

print("\n" + "="*60)
print("CONFUSION MATRIX")
print("="*60)

cm = confusion_matrix(y_test, final_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()


# ════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ════════════════════════════════════════════════════

print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)

importance = final_model.feature_importances_

features = X.columns

importance_df = pd.DataFrame({

    'Feature': features,

    'Importance': importance
})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

print(importance_df.head(10))


plt.figure(figsize=(10,5))

sns.barplot(
    x='Importance',
    y='Feature',
    data=importance_df.head(10)
)

plt.title("Top 10 Important Features")

plt.show()


# ════════════════════════════════════════════════════
# OVERFITTING CHECK
# ════════════════════════════════════════════════════

print("\n" + "="*60)
print("OVERFITTING CHECK")
print("="*60)

train_pred = final_model.predict(X_train)

train_acc = accuracy_score(y_train, train_pred)

train_f1 = f1_score(
    y_train,
    train_pred,
    zero_division=0
)

test_acc = accuracy_score(y_test, final_pred)

test_f1 = f1_score(
    y_test,
    final_pred,
    zero_division=0
)

print(f"{'Metric':<15} {'Train':<12} {'Test':<12}")

print("-"*40)

print(f"{'Accuracy':<15} {train_acc:<12.4f} {test_acc:<12.4f}")

print(f"{'F1-Score':<15} {train_f1:<12.4f} {test_f1:<12.4f}")

difference = abs(train_f1 - test_f1)

print("\nDifference:", round(difference, 4))

if difference < 0.05:

    print("RESULT: No Overfitting Detected")

else:

    print("WARNING: Overfitting Detected")


# ════════════════════════════════════════════════════
# SAVE MODEL
# ════════════════════════════════════════════════════

import os
import joblib

os.makedirs("../ml_models", exist_ok=True)

joblib.dump(
    final_model,
    "../ml_models/fraud_model.pkl"
)

print("✅ Model Saved Successfully!")


# ════════════════════════════════════════════════════
# TEST CUSTOM TRANSACTION
# ════════════════════════════════════════════════════

print("\n" + "="*60)
print("CUSTOM TRANSACTION TEST")
print("="*60)

sample = X_test.iloc[0:1]

prediction = final_model.predict(sample)[0]

probability = final_model.predict_proba(sample)[0][1]

print("Fraud Probability:", round(probability * 100, 2), "%")

if prediction == 1:

    print("🚨 FRAUD DETECTED")

else:

    print("✅ LEGITIMATE TRANSACTION")


# ════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════

print("\n" + "="*60)
print("PROJECT SUMMARY")
print("="*60)

print("✔ Dataset Loaded")

print("✔ Data Visualization Completed")

print("✔ Multiple Models Trained")

print("✔ Accuracy Evaluated")

print("✔ Precision Evaluated")

print("✔ Recall Evaluated")

print("✔ F1-Score Evaluated")

print("✔ ROC-AUC Evaluated")

print("✔ Hyperparameter Tuning Completed")

print("✔ Best Model Selected")

print("✔ Classification Report Generated")

print("✔ Confusion Matrix Generated")

print("✔ Feature Importance Generated")

print("✔ Overfitting Checked")

print("✔ Final Model Saved")

print("✔ Fraud Detection System Completed")

print("\n🚀 Financial Fraud Detection System Ready!")