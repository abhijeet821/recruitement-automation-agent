"""
Django settings.

Security changes from the previous version, each fixing a real defect:

* ``DEBUG`` now defaults to **False**. It previously defaulted to ``True`` while
  the env declaration claimed otherwise, so a deployment that forgot to set the
  variable shipped with the debug console and full tracebacks exposed.
* ``ALLOWED_HOSTS`` no longer contains ``'*'`` by default, which had disabled
  Host-header validation entirely.
* ``SECRET_KEY`` has no insecure fallback in production. A missing key is a
  startup error, not a silent downgrade to a key committed in the repository.
* Security headers (HSTS, referrer policy, cookie flags) are enabled whenever
  DEBUG is off.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import dj_database_url
import environ

BASE_DIR = Path(__file__).resolve().parent.parent

env = environ.Env()
environ.Env.read_env(BASE_DIR / ".env")

# ── Core ─────────────────────────────────────────────────────
DEBUG = env.bool("DEBUG", default=False)

_INSECURE_DEV_KEY = "django-insecure-dev-only-do-not-use-in-production"
SECRET_KEY = env("SECRET_KEY", default="")
if not SECRET_KEY:
    if DEBUG or "test" in sys.argv or "pytest" in sys.modules:
        SECRET_KEY = _INSECURE_DEV_KEY
    else:
        raise environ.ImproperlyConfigured(
            "SECRET_KEY must be set when DEBUG is False. "
            "Generate one with: python -c "
            "'from django.core.management.utils import get_random_secret_key;"
            "print(get_random_secret_key())'"
        )

ALLOWED_HOSTS = env.list("ALLOWED_HOSTS", default=["127.0.0.1", "localhost"])
CSRF_TRUSTED_ORIGINS = env.list(
    "CSRF_TRUSTED_ORIGINS",
    default=["http://127.0.0.1:8000", "http://localhost:8000"],
)

# Encryption key for stored OAuth tokens. Falls back to a key derived from
# SECRET_KEY (see hiring_app/crypto.py) — which means rotating SECRET_KEY forces
# every user to reconnect Google. Set this explicitly to decouple the two.
FIELD_ENCRYPTION_KEY = env("FIELD_ENCRYPTION_KEY", default="")

if DEBUG:
    # Google's OAuth library refuses a plain-HTTP redirect URI unless told the
    # transport is intentionally insecure. Local development only.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

# ── Applications ─────────────────────────────────────────────
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django_ratelimit",
    "hiring_app.apps.HiringAppConfig",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "my_hiring_project.urls"
WSGI_APPLICATION = "my_hiring_project.wsgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
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

# ── Database ─────────────────────────────────────────────────
DATABASES = {
    "default": dj_database_url.config(
        default=f"sqlite:///{BASE_DIR / 'db.sqlite3'}",
        conn_max_age=600,
        conn_health_checks=True,
    )
}

# ── Authentication ───────────────────────────────────────────
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
     "OPTIONS": {"min_length": 8}},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LOGIN_URL = "/login/"
LOGIN_REDIRECT_URL = "/dashboard/"
LOGOUT_REDIRECT_URL = "/"

# ── Sessions & security headers ──────────────────────────────
SESSION_COOKIE_AGE = 86_400
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"
CSRF_COOKIE_SAMESITE = "Lax"
SESSION_COOKIE_SECURE = not DEBUG
CSRF_COOKIE_SECURE = not DEBUG

SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_REFERRER_POLICY = "same-origin"
X_FRAME_OPTIONS = "DENY"

if not DEBUG:
    SECURE_SSL_REDIRECT = env.bool("SECURE_SSL_REDIRECT", default=True)
    # Terminating proxies (Railway, Heroku, most PaaS) forward the original
    # scheme in this header; without it Django sees HTTP and redirect-loops.
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
    SECURE_HSTS_SECONDS = env.int("SECURE_HSTS_SECONDS", default=31_536_000)
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True

# ── Internationalisation ─────────────────────────────────────
LANGUAGE_CODE = "en-us"
TIME_ZONE = env("TIME_ZONE", default="Asia/Kolkata")
USE_I18N = True
USE_TZ = True

# ── Static & media ───────────────────────────────────────────
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_DIRS = [BASE_DIR / "hiring_app" / "static"]

MEDIA_URL = "/media/"
MEDIA_ROOT = Path(env("MEDIA_ROOT", default=str(BASE_DIR / "media")))

TESTING = "pytest" in sys.modules or "test" in sys.argv

STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    # The manifest backend requires `collectstatic` to have run, which is right
    # for production and wrong for a test run — under test it makes every page
    # render fail with a missing-manifest error unrelated to what is being tested.
    "staticfiles": {
        "BACKEND": (
            "django.contrib.staticfiles.storage.StaticFilesStorage"
            if TESTING
            else "whitenoise.storage.CompressedManifestStaticFilesStorage"
        )
    },
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
SILENCED_SYSTEM_CHECKS = ["django_ratelimit.E003", "django_ratelimit.W001"]

# ── Matching engine ──────────────────────────────────────────
# Parsed here and pushed into the engine so the pure-Python package needs no
# Django import while still honouring one source of configuration.
LLM_PROVIDER = env("LLM_PROVIDER", default="ollama")
OLLAMA_HOST = env("OLLAMA_HOST", default="http://localhost:11434")
OLLAMA_MODEL = env("OLLAMA_MODEL", default="qwen3:14b")
OLLAMA_EMBED_MODEL = env("OLLAMA_EMBED_MODEL", default="bge-m3")
GEMINI_API_KEY = env("GEMINI_API_KEY", default="")
GEMINI_MODEL = env("GEMINI_MODEL", default="gemini-2.5-flash")
GITHUB_TOKEN = env("GITHUB_TOKEN", default="")
BLIND_SCREENING = env.bool("BLIND_SCREENING", default=True)
RUBRIC_ENABLED = env.bool("RUBRIC_ENABLED", default=True)
GOOGLE_CLIENT_SECRETS = env("GOOGLE_CLIENT_SECRETS", default="")


def _configure_matching_engine() -> None:
    from matching.config import MatchingConfig, set_config

    set_config(MatchingConfig(
        provider=LLM_PROVIDER,
        ollama_host=OLLAMA_HOST,
        ollama_model=OLLAMA_MODEL,
        ollama_embed_model=OLLAMA_EMBED_MODEL,
        gemini_api_key=GEMINI_API_KEY,
        gemini_model=GEMINI_MODEL,
        cache_dir=Path(env("MATCHING_CACHE_DIR", default=str(BASE_DIR / ".cache"))),
        cache_enabled=env.bool("MATCHING_CACHE", default=True),
        github_token=GITHUB_TOKEN,
        blind_screening=BLIND_SCREENING,
        rubric_enabled=RUBRIC_ENABLED,
        request_timeout=env.float("LLM_TIMEOUT", default=180.0),
    ))


_configure_matching_engine()

# ── Logging ──────────────────────────────────────────────────
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {"format": "[{levelname}] {asctime} {name} — {message}", "style": "{"},
        "simple": {"format": "[{levelname}] {message}", "style": "{"},
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "simple"},
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": LOGS_DIR / "app.log",
            "maxBytes": 5 * 1024 * 1024,
            "backupCount": 3,
            "formatter": "verbose",
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": LOGS_DIR / "errors.log",
            "maxBytes": 5 * 1024 * 1024,
            "backupCount": 3,
            "level": "ERROR",
            "formatter": "verbose",
        },
    },
    "loggers": {
        "django": {"handlers": ["console", "file"], "level": "INFO"},
        "django.request": {"handlers": ["error_file", "console"], "level": "ERROR",
                           "propagate": False},
        "hiring_app": {"handlers": ["console", "file", "error_file"],
                       "level": env("LOG_LEVEL", default="INFO"), "propagate": False},
        "matching": {"handlers": ["console", "file", "error_file"],
                     "level": env("LOG_LEVEL", default="INFO"), "propagate": False},
    },
}

# ── Email (error notifications only) ─────────────────────────
ADMIN_EMAIL = env("ADMIN_EMAIL", default="")
if ADMIN_EMAIL:
    ADMINS = [("Admin", ADMIN_EMAIL)]

if DEBUG:
    EMAIL_BACKEND = "django.core.mail.backends.console.EmailBackend"
else:
    EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"
    EMAIL_HOST = env("EMAIL_HOST", default="smtp.gmail.com")
    EMAIL_PORT = env.int("EMAIL_PORT", default=587)
    EMAIL_USE_TLS = True
    EMAIL_HOST_USER = env("EMAIL_HOST_USER", default="")
    EMAIL_HOST_PASSWORD = env("EMAIL_HOST_PASSWORD", default="")
    SERVER_EMAIL = env("EMAIL_HOST_USER", default="errors@localhost")
