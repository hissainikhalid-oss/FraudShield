import os
import django

# Setup Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "your_project_name.settings")
django.setup()

from fraudshield.ml_utils import predict_fraud
from fraudshield.ai_utils import analyze_transaction_with_ai


def test_pipeline():
    print("\n🔍 Testing Fraud Detection Pipeline...\n")

    # Sample test transaction
    amount = 85000
    source = "Mobile"

    print(f"Amount: ₹{amount}")
    print(f"Source: {source}")

    # Step 1 — ML Prediction
    is_fraud, prob, risk_score = predict_fraud(amount, source)

    print("\n📊 ML Model Output")
    print("Is Fraud:", is_fraud)
    print("Probability:", prob)
    print("Risk Score:", risk_score)

    # Step 2 — AI Analysis
    ai_result = analyze_transaction_with_ai(
        amount=amount,
        source=source,
        time_period='afternoon',
        location='Home city',
        frequency='First transaction',
        sender_name='Khalid',
        receiver_name='Unknown',
        receiver_rel='Unknown',
        ml_risk_score=risk_score,
        ml_prediction=is_fraud
    )

    # Step 3 — Final Risk Calculation
    risk = risk_score
    ai_adjust = (ai_result['riskScore'] - 50) * 0.2
    risk = max(0, min(100, int(risk + ai_adjust)))

    if risk >= 75:
        result = '🚨 Fraud Detected'
    elif risk >= 40:
        result = '⚠ Suspicious Transaction'
    else:
        result = '✅ Safe Transaction'

    print("\n🤖 AI Analysis")
    print("AI Risk:", ai_result['riskScore'])
    print("Explanation:", ai_result['explanation'])

    print("\n✅ Final Decision")
    print("Final Risk:", risk)
    print("Result:", result)


if __name__ == "__main__":
    test_pipeline()