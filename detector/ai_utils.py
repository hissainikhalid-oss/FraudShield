"""
AI utilities using Groq API with LLaMA 3.1 for explanations.
"""
import os
import json
import logging
from groq import Groq
from django.conf import settings

logger = logging.getLogger(__name__)


def get_groq_client():
    """Initialize Groq client with API key."""
    api_key = getattr(settings, "GROQ_API_KEY", None) or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Add it to your environment or Django settings."
        )
    return Groq(api_key=api_key)


def analyze_transaction_with_ai(
    amount,
    source,
    time_period,
    location,
    frequency,
    sender_name,
    receiver_name,
    receiver_rel,
    ml_risk_score,
    ml_prediction,
    user_context=None
):

    prompt = f"""You are an expert financial fraud analyst AI. Analyze this transaction and provide your assessment.

TRANSACTION DETAILS:
- Amount: ₹{amount:,.2f}
- Source: {source}
- Time Period: {time_period}
- Location: {location}
- Transaction Frequency: {frequency}
- Sender: {sender_name}
- Receiver: {receiver_name}
- Relationship with Receiver: {receiver_rel}

MACHINE LEARNING MODEL OUTPUT:
- ML Prediction: {"FRAUDULENT" if ml_prediction else "LEGITIMATE"}
- ML Risk Score: {ml_risk_score:.2f}%

{f"ADDITIONAL USER CONTEXT: {user_context}" if user_context else ""}

IMPORTANT: Respond ONLY with valid JSON in this exact format:
{{
    "verdict": "FRAUD" or "SAFE",
    "summary": "One sentence summary",
    "riskScore": <0-100>,
    "factors": [],
    "explanation": "short explanation"
}}
"""

    try:
        client = get_groq_client()

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Respond with JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800,
        )

        content = response.choices[0].message.content.strip()

        # extract JSON safely
        start = content.find("{")
        end = content.rfind("}")
        content = content[start:end + 1]

        result = json.loads(content)

        # ---- FIXED NORMALIZATION ----
        try:
            risk_score = float(result.get("riskScore", ml_risk_score))
        except:
            risk_score = ml_risk_score

        # convert 0-1 → 0-100
        if risk_score <= 1:
            risk_score *= 100

        risk_score = max(0, min(100, int(risk_score)))

        return {
            "verdict": result.get("verdict", "SAFE"),
            "summary": result.get("summary", "Transaction analyzed."),
            "riskScore": risk_score,
            "factors": result.get("factors", []),
            "explanation": result.get("explanation", "No explanation available.")
        }

    except Exception:
        return create_fallback_response(amount, source, ml_risk_score, ml_prediction)


def create_fallback_response(amount, source, ml_risk_score, ml_prediction):

    risk = ml_risk_score
    if risk <= 1:
        risk *= 100

    risk = max(0, min(100, int(risk)))

    return {
        "verdict": "FRAUD" if ml_prediction else "SAFE",
        "summary": "Fallback ML-based analysis.",
        "riskScore": risk,
        "factors": [
            {
                "name": "Amount",
                "level": "high" if amount > 10000 else "low",
                "description": f"Transaction amount ₹{amount:,.2f}"
            },
            {
                "name": "Source",
                "level": "medium",
                "description": f"Source: {source}"
            }
        ],
        "explanation": f"The ML model predicted {'fraud' if ml_prediction else 'safe'} with risk {risk}%."
    }


def chat_with_ai(messages):

    try:
        client = get_groq_client()

        if not any(m.get('role') == 'system' for m in messages):
            messages.insert(0, {
                "role": "system",
                "content": "You are FraudShield AI Assistant."
            })

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.7,
            max_tokens=300,
        )

        return response.choices[0].message.content.strip()

    except Exception:
        return "AI connection error. Please try again."