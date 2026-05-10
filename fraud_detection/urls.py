# fraud_detection/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('detector.urls')),  # include your app’s urls
]