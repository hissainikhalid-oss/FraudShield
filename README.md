# Financial Fraud Detection System

An intelligent machine learning system that detects fraudulent financial transactions in real time, with an AI-powered explanation engine and an interactive web dashboard.

---

## Project Structure

```
fraud_detection/
├── data/
│   ├── raw/                        # Original dataset (creditcard.csv)
│   └── processed/                  # Cleaned and feature-engineered data
├── notebooks/
│   └── eda.ipynb                   # Exploratory Data Analysis notebook
├── src/
│   ├── data_preprocessing.py       # Load, clean, encode data
│   ├── feature_engineering.py      # Create new features, scale data
│   ├── train.py                    # Train ML models, save best model
│   ├── evaluate.py                 # Evaluate model, plot ROC curve
│   └── predict.py                  # Predict fraud for single transaction
├── models/
│   ├── fraud_model.pkl             # Saved Random Forest model
│   └── scaler.pkl                  # Saved StandardScaler
├── app/
│   └── app.py                      # Web application entry point
├── requirements.txt
├── README.md
└── main.py                         # Run full ML pipeline
```

---

## ML Models Used

| Model | F1-Score | AUC-ROC |
|---|---|---|
| Logistic Regression | 0.7500 | 0.9910 |
| Decision Tree | 0.9804 | 0.9990 |
| **Random Forest** ✓ | **0.9933** | **1.0000** |

---

## Features

- Real-time fraud prediction using Random Forest
- AI-powered explanation of each transaction (Groq LLaMA)
- Sender & Receiver details tracking
- Chat with AI about any transaction
- False positive challenge — user can dispute wrong results
- Interactive Bootstrap dashboard with charts

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place dataset in data/raw/
# Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# 3. Run full ML pipeline
python main.py

# 4. Start web dashboard
python manage.py runserver
```

---

## Dataset

- **Source**: ULB Credit Card Fraud Detection (Kaggle)
- **Size**: 284,807 transactions
- **Fraud rate**: 0.17% (highly imbalanced)
- **Features**: V1–V28 (PCA), Time, Amount, Class

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Django, Python |
| ML | Scikit-learn, Random Forest |
| AI Explanation | Groq API, LLaMA 3.1 |
| Frontend | HTML, Bootstrap 5, Chart.js |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |

---

*Developed as part of DS/AI Internship — March 2026*
# FraudShield
# FraudShield
