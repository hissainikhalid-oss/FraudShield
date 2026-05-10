"""
Machine Learning utilities for fraud detection.
"""

import os
import joblib
import numpy as np
from pathlib import Path


# =====================================================
# MODEL PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / 'ml_model'

MODEL_PATH = MODEL_DIR / 'fraud_model.pkl'

SCALER_PATH = MODEL_DIR / 'scaler.pkl'

TIME_SCALER_PATH = MODEL_DIR / 'time_scaler.pkl'


# =====================================================
# GLOBAL VARIABLES
# =====================================================

_model = None

_scaler = None

_time_scaler = None


# =====================================================
# LOAD MODEL
# =====================================================

def get_model():

    global _model

    if _model is None:

        _model = joblib.load(MODEL_PATH)

        print("✅ Fraud Detection Model Loaded")

    return _model


# =====================================================
# LOAD SCALERS
# =====================================================

def get_scaler():

    global _scaler

    if _scaler is None:

        if SCALER_PATH.exists():

            _scaler = joblib.load(SCALER_PATH)

    return _scaler


def get_time_scaler():

    global _time_scaler

    if _time_scaler is None:

        if TIME_SCALER_PATH.exists():

            _time_scaler = joblib.load(TIME_SCALER_PATH)

    return _time_scaler


# =====================================================
# FEATURE ENGINEERING
# =====================================================

def preprocess_transaction(

    amount,

    source,

    time_period='afternoon',

    location='home',

    frequency='normal'

):

    features = np.zeros(30)

    # -----------------------------------------
    # TIME FEATURE
    # -----------------------------------------

    time_mapping = {

        'morning': 9 * 3600,

        'afternoon': 15 * 3600,

        'evening': 20 * 3600,

        'night': 2 * 3600,
    }

    time_seconds = time_mapping.get(
        time_period.lower(),
        15 * 3600
    )

    try:

        time_scaler = get_time_scaler()

        if time_scaler:

            features[0] = time_scaler.transform(
                [[time_seconds]]
            )[0][0]

        else:

            features[0] = time_seconds / 172800

    except:

        features[0] = time_seconds / 172800


    # -----------------------------------------
    # AMOUNT FEATURE
    # -----------------------------------------

    try:

        scaler = get_scaler()

        if scaler:

            features[29] = scaler.transform(
                [[amount]]
            )[0][0]

        else:

            features[29] = np.log1p(amount) / 10

    except:

        features[29] = np.log1p(amount) / 10


    # -----------------------------------------
    # SOURCE RISK
    # -----------------------------------------

    source_risk = {

        'web': 0.15,

        'mobile': 0.20,

        'atm': 0.35,

        'pos': 0.10
    }

    base_risk = source_risk.get(
        source.lower(),
        0.20
    )


    # -----------------------------------------
    # AMOUNT RISK
    # -----------------------------------------

    if amount > 100000:

        base_risk += 0.40

    elif amount > 50000:

        base_risk += 0.30

    elif amount > 10000:

        base_risk += 0.20

    elif amount > 5000:

        base_risk += 0.10


    # -----------------------------------------
    # NIGHT RISK
    # -----------------------------------------

    if time_period.lower() == 'night':

        base_risk += 0.15


    # -----------------------------------------
    # INTERNATIONAL RISK
    # -----------------------------------------

    if location.lower() == 'international':

        base_risk += 0.20


    # -----------------------------------------
    # HIGH FREQUENCY RISK
    # -----------------------------------------

    if frequency.lower() == 'high':

        base_risk += 0.15


    # -----------------------------------------
    # SYNTHETIC PCA FEATURES
    # -----------------------------------------

    np.random.seed(int(amount) % 10000)

    for i in range(1, 29):

        noise = np.random.uniform(-0.5, 0.5)

        if i <= 14:

            features[i] = (
                -2 + base_risk * 4 + noise
            )

        else:

            features[i] = (
                2 - base_risk * 4 + noise
            )

    return features.reshape(1, -1)


# =====================================================
# FRAUD PREDICTION
# =====================================================

def predict_fraud(

    amount,

    source='web',

    time_period='afternoon',

    location='home',

    frequency='normal'

):

    model = get_model()

    features = preprocess_transaction(

        amount,

        source,

        time_period,

        location,

        frequency
    )

    probabilities = model.predict_proba(features)[0]

    fraud_probability = probabilities[1]


    # -----------------------------------------
    # BASE RISK FROM AMOUNT
    # -----------------------------------------

    if amount < 500:

        base_risk = 10

    elif amount < 2000:

        base_risk = 20

    elif amount < 10000:

        base_risk = 40

    elif amount < 50000:

        base_risk = 60

    else:

        base_risk = 80


    # -----------------------------------------
    # ML ADJUSTMENT
    # -----------------------------------------

    ml_adjustment = fraud_probability * 20

    risk_score = base_risk + ml_adjustment

    risk_score = max(1, min(99, risk_score))


    # -----------------------------------------
    # FINAL DECISION
    # -----------------------------------------

    is_fraud = risk_score >= 70

    return (

        bool(is_fraud),

        round(float(fraud_probability), 4),

        round(float(risk_score), 2)
    )