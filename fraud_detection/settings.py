"""
Django settings for fraud_detection project.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# ---------------------------------------------------
# LOAD ENV VARIABLES
# ---------------------------------------------------

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------
# SECURITY
# ---------------------------------------------------

SECRET_KEY = os.getenv(
    "SECRET_KEY",
    "django-insecure-dev-key"
)

DEBUG = False

ALLOWED_HOSTS = [
    "fraudsystem.pythonanywhere.com",
    "127.0.0.1",
    "localhost"
]

# ---------------------------------------------------
# APPLICATIONS
# ---------------------------------------------------

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",

    "detector",
]

# ---------------------------------------------------
# MIDDLEWARE
# ---------------------------------------------------

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",

    "django.contrib.sessions.middleware.SessionMiddleware",

    "django.middleware.common.CommonMiddleware",

    "django.middleware.csrf.CsrfViewMiddleware",

    "django.contrib.auth.middleware.AuthenticationMiddleware",

    "django.contrib.messages.middleware.MessageMiddleware",

    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

# ---------------------------------------------------
# URLS
# ---------------------------------------------------

ROOT_URLCONF = "fraud_detection.urls"

# ---------------------------------------------------
# TEMPLATES
# ---------------------------------------------------

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",

        "DIRS": [
            BASE_DIR / "detector/templates"
        ],

        "APP_DIRS": True,

        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",

                "django.contrib.auth.context_processors.auth",

                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

# ---------------------------------------------------
# WSGI
# ---------------------------------------------------

WSGI_APPLICATION = "fraud_detection.wsgi.application"

# ---------------------------------------------------
# DATABASE
# ---------------------------------------------------

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",

        "NAME": BASE_DIR / "db.sqlite3",
    }
}

# ---------------------------------------------------
# PASSWORD VALIDATION
# ---------------------------------------------------

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME":
        "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },

    {
        "NAME":
        "django.contrib.auth.password_validation.MinimumLengthValidator",
    },

    {
        "NAME":
        "django.contrib.auth.password_validation.CommonPasswordValidator",
    },

    {
        "NAME":
        "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

# ---------------------------------------------------
# LANGUAGE / TIMEZONE
# ---------------------------------------------------

LANGUAGE_CODE = "en-us"

TIME_ZONE = "UTC"

USE_I18N = True

USE_TZ = True

# ---------------------------------------------------
# STATIC FILES
# ---------------------------------------------------

STATIC_URL = "/static/"

STATICFILES_DIRS = [
    BASE_DIR / "static"
]

STATIC_ROOT = BASE_DIR / "staticfiles"

# ---------------------------------------------------
# DEFAULT PRIMARY KEY
# ---------------------------------------------------

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ---------------------------------------------------
# GROQ API KEY
# ---------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---------------------------------------------------
# SECURITY SETTINGS FOR DEPLOYMENT
# ---------------------------------------------------

CSRF_TRUSTED_ORIGINS = [
    "https://fraudsystem.pythonanywhere.com"
]

SECURE_BROWSER_XSS_FILTER = True

X_FRAME_OPTIONS = "DENY"
