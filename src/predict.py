import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "fraud_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

print("Looking for model at:", model_path)
print("Looking for scaler at:", scaler_path)

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

print("SUCCESS: Model and scaler loaded!")
