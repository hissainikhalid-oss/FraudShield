import numpy as np
import joblib

# Load model
model = joblib.load("fraud_model.pkl")

print(f"[INFO] Model loaded: {type(model).__name__}")


def predict_transaction(transaction: dict) -> dict:

    amount = transaction.get("Amount", 0)
    time = transaction.get("Time", 0)

    # Feature engineering
    hour = int((time % 86400) // 3600)
    is_night = 1 if (hour < 6 or hour > 22) else 0
    amount_log = np.log1p(amount)
    amount_high = 1 if amount > 10000 else 0

    transaction["Hour"] = hour
    transaction["Is_Night"] = is_night
    transaction["Amount_Log"] = amount_log
    transaction["Amount_High"] = amount_high

    # 34 features
    feature_order = [
        "V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
        "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
        "V21","V22","V23","V24","V25","V26","V27","V28",
        "Amount","Time"
]

    # Fill missing
    for f in feature_order:
        if f not in transaction:
            transaction[f] = 0

    features = np.array([transaction[f] for f in feature_order]).reshape(1, -1)

    # Predict
    probability = model.predict_proba(features)[0][1]
    risk_score = probability * 100

    # Base risk from ML model
    risk_score = probability * 100

    # Amount-based adjustments
    if amount > 100000:
        risk_score += 30
    elif amount > 50000:
        risk_score += 15

    # Final decision (single place)  
    is_fraud = risk_score >= 25

    result = {
        "verdict": "FRAUD" if is_fraud else "SAFE",
        "probability": round(float(probability), 4),
        "risk_score": round(float(risk_score), 2),
}
    print(f"\n[RESULT] Verdict    : {result['verdict']}")
    print(f"[RESULT] Probability: {result['probability']}")
    print(f"[RESULT] Risk Score : {result['risk_score']}%")

    return result


if __name__ == "__main__":
    sample_transaction = {
        "Amount": 120000,
        "Time": 0.0
    }

    predict_transaction(sample_transaction)
