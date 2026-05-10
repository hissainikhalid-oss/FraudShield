"""
Views for FraudShield fraud detection system.
"""

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET
from django.db.models import Count
import json

from .models import Transaction
from .ml_utils import predict_fraud
from .ai_utils import analyze_transaction_with_ai, chat_with_ai


# =====================================================
# HOME PAGE
# =====================================================

def home(request):

    transactions = Transaction.objects.all().order_by('-id')

    fraud_count = transactions.filter(
        risk__gte=70
    ).count()

    safe_count = transactions.filter(
        risk__lt=40
    ).count()

    transactions = transactions[:50]

    # -------------------------------------------------
    # SOURCE CHART DATA
    # -------------------------------------------------

    source_counts = Transaction.objects.values(
        'source'
    ).annotate(
        count=Count('source')
    )

    source_data = {
        'Web': 0,
        'Mobile': 0,
        'ATM': 0,
        'POS': 0
    }

    for item in source_counts:

        source_data[item['source']] = item['count']

    result = None
    risk = None
    explanation = None

    # -------------------------------------------------
    # FORM SUBMISSION
    # -------------------------------------------------

    if request.method == 'POST':

        try:

            amount = float(
                request.POST.get('amount', 0)
            )

            source = request.POST.get(
                'source',
                'Web'
            )

            sender_name = request.POST.get(
                'sender_name',
                'Unknown'
            )

            sender_acc = request.POST.get(
                'sender_acc',
                'XXXX'
            )

            receiver_name = request.POST.get(
                'receiver_name',
                'Unknown'
            )

            receiver_rel = request.POST.get(
                'receiver_rel',
                'Unknown'
            )

            # -----------------------------------------
            # ML PREDICTION
            # -----------------------------------------

            is_fraud, prob, risk_score = predict_fraud(
                amount=amount,
                source=source,
                time_period='afternoon',
                location='home',
                frequency='normal'
            )

            # -----------------------------------------
            # AI ANALYSIS
            # -----------------------------------------

            ai_result = analyze_transaction_with_ai(
                amount=amount,
                source=source,
                time_period='afternoon',
                location='Home city',
                frequency='normal',
                sender_name=sender_name,
                receiver_name=receiver_name,
                receiver_rel=receiver_rel,
                ml_risk_score=risk_score,
                ml_prediction=is_fraud
            )

            # -----------------------------------------
            # FINAL RISK
            # -----------------------------------------

            risk = int(
                ai_result.get(
                    'riskScore',
                    risk_score
                )
            )

            risk = max(
                0,
                min(100, risk)
            )

            # -----------------------------------------
            # FINAL STATUS
            # -----------------------------------------

            if risk >= 70:

                result = '🚨 Fraud Detected'

            elif risk >= 40:

                result = '⚠ Suspicious Transaction'

            else:

                result = '✅ Safe Transaction'

            explanation = ai_result.get(
                'explanation'
            )

            # -----------------------------------------
            # SAVE TRANSACTION
            # -----------------------------------------

            Transaction.objects.create(
                amount=amount,
                source=source,
                risk=risk,
                result=result,
                explanation=explanation,
                sender_name=sender_name,
                sender_acc=sender_acc,
                receiver_name=receiver_name,
                receiver_rel=receiver_rel
            )

        except Exception as e:

            result = 'Error'
            risk = 0
            explanation = str(e)

    context = {
        'transactions': transactions,
        'fraud_count': fraud_count,
        'safe_count': safe_count,
        'result': result,
        'risk': risk,
        'explanation': explanation,
        'source_data': source_data,
    }

    return render(
        request,
        'index.html',
        context | {'active_page': 'dashboard'}
    )


# =====================================================
# AI ANALYSIS API
# =====================================================

@csrf_exempt
@require_POST
def api_ai_analyze(request):

    try:

        data = json.loads(request.body)

        amount = float(data.get('amount', 0))

        source = data.get(
            'source',
            'Web'
        )

        time_period = data.get(
            'time',
            'afternoon'
        )

        location = data.get(
            'location',
            'home'
        )

        frequency = data.get(
            'frequency',
            'normal'
        )

        sender_name = data.get(
            'senderName',
            'Unknown'
        )

        receiver_name = data.get(
            'receiverName',
            'Unknown'
        )

        receiver_rel = data.get(
            'receiverRel',
            'Unknown'
        )

        user_context = data.get(
            'userContext',
            None
        )

        # -----------------------------------------
        # ML PREDICTION
        # -----------------------------------------

        is_fraud, prob, risk_score = predict_fraud(
            amount=amount,
            source=source,
            time_period=time_period,
            location=location,
            frequency=frequency
        )

        # -----------------------------------------
        # AI ANALYSIS
        # -----------------------------------------

        ai_result = analyze_transaction_with_ai(
            amount=amount,
            source=source,
            time_period=time_period,
            location=location,
            frequency=frequency,
            sender_name=sender_name,
            receiver_name=receiver_name,
            receiver_rel=receiver_rel,
            ml_risk_score=risk_score,
            ml_prediction=is_fraud,
            user_context=user_context
        )

        # -----------------------------------------
        # FINAL RISK
        # -----------------------------------------

        risk = int(
            ai_result.get(
                'riskScore',
                risk_score
            )
        )

        risk = max(
            0,
            min(100, risk)
        )

        # -----------------------------------------
        # FINAL STATUS
        # -----------------------------------------

        if risk >= 70:

            status = '🚨 Fraud Detected'

        elif risk >= 40:

            status = '⚠ Suspicious Transaction'

        else:

            status = '✅ Safe Transaction'

        return JsonResponse({

            'riskScore': risk,

            'verdict': (
                'FRAUD'
                if risk >= 70
                else 'SAFE'
            ),

            'summary': (
                'High fraud probability detected.'
                if risk >= 70
                else 'Transaction appears normal.'
            ),

            'factors': [

                {
                    'name': 'Transaction Amount',

                    'level': (
                        'high'
                        if amount > 50000
                        else 'medium'
                        if amount > 10000
                        else 'low'
                    ),

                    'description': (
                        f'Transaction amount ₹{amount}'
                    )
                },

                {
                    'name': 'Transaction Source',

                    'level': (
                        'high'
                        if source.lower() == 'atm'
                        else 'medium'
                    ),

                    'description': (
                        f'Source: {source}'
                    )
                }

            ],

            'confidence': round(
                prob * 100,
                2
            ),

            'explanation': ai_result.get(
                'explanation'
            )
        })

    except Exception as e:

        return JsonResponse(
            {'error': str(e)},
            status=500
        )


# =====================================================
# AI CHAT API
# =====================================================

@csrf_exempt
@require_POST
def api_ai_chat(request):

    try:

        data = json.loads(request.body)

        messages = data.get(
            'messages',
            []
        )

        if not messages:

            return JsonResponse({
                "reply": "Hello! I'm FraudShield AI."
            })

        reply = chat_with_ai(messages)

        reply = reply or (
            "Ask me about fraud detection, "
            "risk scores, or suspicious transactions."
        )

        return JsonResponse({
            "reply": reply
        })

    except Exception as e:

        return JsonResponse({
            "reply": f"AI server error: {str(e)}"
        })


# =====================================================
# STATS API
# =====================================================

@require_GET
def api_stats(request):

    total = Transaction.objects.count()

    fraud_count = Transaction.objects.filter(
        risk__gte=70
    ).count()

    safe_count = Transaction.objects.filter(
        risk__lt=40
    ).count()

    return JsonResponse({
        'total': total,
        'fraud': fraud_count,
        'safe': safe_count
    })