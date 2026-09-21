"""
Django settings for orbit_predictor project.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/5.2/ref/settings/
"""

import datetime
import os
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


def env_bool(name, default=False):
    """Read a boolean from the environment ('1', 'true', 'yes', 'on')."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_list(name, default=None):
    """Read a comma-separated list from the environment."""
    raw = os.getenv(name)
    if not raw:
        return list(default or [])
    return [item.strip() for item in raw.split(",") if item.strip()]


# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = env_bool("DEBUG", default=False)

# SECURITY WARNING: keep the secret key used in production secret!
# This used to be a literal checked into source control, which meant every
# deployment shared a publicly-known key and could have session/signing data
# forged against it. It now comes from the environment; an insecure
# development-only fallback is generated when DEBUG is on so that a fresh
# checkout still runs without setup.
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    if DEBUG:
        from django.core.management.utils import get_random_secret_key

        SECRET_KEY = get_random_secret_key()
    else:
        raise ImproperlyConfigured(
            "SECRET_KEY must be set in the environment when DEBUG is off. "
            "Generate one with: python -c "
            "'from django.core.management.utils import get_random_secret_key;"
            "print(get_random_secret_key())'"
        )

# Hosts this app will serve. Defaults to localhost for development; set
# ALLOWED_HOSTS in the environment (comma separated) for any real deployment.
ALLOWED_HOSTS = env_list(
    "ALLOWED_HOSTS",
    default=["localhost", "127.0.0.1", "[::1]"] if DEBUG else [],
)
if not DEBUG and not ALLOWED_HOSTS:
    raise ImproperlyConfigured(
        "ALLOWED_HOSTS must be set in the environment when DEBUG is off."
    )


# Application definition
#
# Note on what is deliberately absent: this project authenticates exclusively
# with stateless JWT bearer tokens (see api.authentication.JWTAuthentication).
# It issues no session cookie, so SessionMiddleware, AuthenticationMiddleware,
# CsrfViewMiddleware and django.contrib.admin are intentionally not enabled --
# CSRF protection guards cookie-borne credentials, which this API does not use.
# If you add cookie-based auth or the Django admin, re-enable all four together.

INSTALLED_APPS = [
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.staticfiles',
    'rest_framework',
    'django_filters',  # Added for filtering
    'corsheaders',     # Added for handling CORS
    'api',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'corsheaders.middleware.CorsMiddleware',  # Must be placed near the top
    'django.middleware.common.CommonMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'orbit_predictor.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
            ],
        },
    },
]

WSGI_APPLICATION = 'orbit_predictor.wsgi.application'


# Database Configuration
#
# PostgreSQL is the deployment target. SQLite is supported for local
# development and for the test suite so that a fresh checkout can run without
# standing up a database server -- set DB_ENGINE=sqlite to use it.
if os.getenv("DB_ENGINE", "postgresql").lower() in {"sqlite", "sqlite3"}:
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': BASE_DIR / 'db.sqlite3',
        }
    }
else:
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.postgresql',
            'NAME': os.getenv('DB_NAME', 'orbit_predictor'),
            'USER': os.getenv('DB_USER', 'postgres'),
            'PASSWORD': os.getenv('DB_PASSWORD', ''),
            'HOST': os.getenv('DB_HOST', 'localhost'),
            'PORT': os.getenv('DB_PORT', '5432'),
        }
    }


ADMIN_REGISTRATION_CODE = os.getenv('ADMIN_REGISTRATION_CODE')

# REST Framework Configuration
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'api.authentication.JWTAuthentication',  # Your custom JWT Authentication
    ),
    'DEFAULT_PERMISSION_CLASSES': (
        'rest_framework.permissions.IsAuthenticated',  # Default to authenticated
    ),
    'DEFAULT_FILTER_BACKENDS': (
        'django_filters.rest_framework.DjangoFilterBackend',
    ),
    # Override these defaults to prevent DRF from using Django's auth system
    'UNAUTHENTICATED_USER': None,
    'UNAUTHENTICATED_TOKEN': None,
}

# JWT signing key. Like SECRET_KEY this must come from the environment; a
# development fallback keeps a fresh checkout runnable but is never used when
# DEBUG is off.
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY')
if not JWT_SECRET_KEY:
    if DEBUG:
        JWT_SECRET_KEY = SECRET_KEY
    else:
        raise ImproperlyConfigured(
            "JWT_SECRET_KEY must be set in the environment when DEBUG is off."
        )

JWT_ALGORITHM = 'HS256'
JWT_ACCESS_EXPIRATION_DELTA = datetime.timedelta(hours=24)  # Access token valid for 24 hours
JWT_REFRESH_EXPIRATION_DELTA = datetime.timedelta(days=7)    # Refresh token valid for 7 days

AUTH_USER_MODEL = 'api.User'


# Internationalization
LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Email Configuration
EMAIL_BACKEND = os.getenv(
    'EMAIL_BACKEND',
    'django.core.mail.backends.console.EmailBackend' if DEBUG
    else 'django.core.mail.backends.smtp.EmailBackend',
)
EMAIL_HOST = os.getenv('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = int(os.getenv('EMAIL_PORT', '587'))
EMAIL_USE_TLS = env_bool('EMAIL_USE_TLS', default=True)
EMAIL_HOST_USER = os.getenv('EMAIL_HOST_USER')
# Strip non-breaking spaces, which app-password values pasted from a browser
# frequently carry and which the SMTP library rejects.
EMAIL_HOST_PASSWORD = os.environ.get("EMAIL_HOST_PASSWORD", "").replace('\xa0', '')


# CORS configuration. Defaults to the local Next.js dev server; override with
# CORS_ALLOWED_ORIGINS (comma separated) for other environments.
CORS_ALLOWED_ORIGINS = env_list(
    "CORS_ALLOWED_ORIGINS",
    default=["http://localhost:3000"],
)


# Security hardening. These only bite when DEBUG is off, so local development
# over plain HTTP is unaffected.
if not DEBUG:
    SECURE_SSL_REDIRECT = env_bool("SECURE_SSL_REDIRECT", default=True)
    SECURE_HSTS_SECONDS = int(os.getenv("SECURE_HSTS_SECONDS", str(60 * 60 * 24 * 365)))
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True
    SECURE_CONTENT_TYPE_NOSNIFF = True
    SECURE_REFERRER_POLICY = "same-origin"
    X_FRAME_OPTIONS = "DENY"
    # Trust X-Forwarded-Proto when running behind a TLS-terminating proxy.
    if env_bool("USE_X_FORWARDED_PROTO", default=True):
        SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")


LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "{asctime} {levelname} {name}: {message}",
            "style": "{",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": os.getenv("LOG_LEVEL", "DEBUG" if DEBUG else "INFO"),
    },
}
