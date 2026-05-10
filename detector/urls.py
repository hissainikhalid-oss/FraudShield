# detector/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('api/ai-analyze/', views.api_ai_analyze, name='api_ai_analyze'),
    path('api/ai-chat/', views.api_ai_chat, name='api_ai_chat'),
    path('api/stats/', views.api_stats, name='api_stats'),
]