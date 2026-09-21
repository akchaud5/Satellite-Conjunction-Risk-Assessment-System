# Build context is the repository root, not this directory -- requirements.txt
# lives at the root. The old Dockerfile did `COPY requirements.txt .` against a
# ./Orbit_Predictor-BackEnd context, where that file does not exist, so the
# image never built. See docker-compose.yml for the matching context/dockerfile
# pair.
#
# Python is pinned to 3.12 because MATLAB Engine for Python supports no higher.
# The engine is not installed here: it ships with MATLAB and cannot be fetched
# in a container. The analytic Pc endpoints return 503 in this image; every
# other endpoint works. See requirements-matlab.txt.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Build dependencies for psycopg2 and the scientific stack.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first so the dependency layer is cached across code changes.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Application code.
COPY Orbit_Predictor-BackEnd/ /app/

EXPOSE 8000

# runserver is a development server. For production, run gunicorn against
# orbit_predictor.wsgi:application behind a real web server.
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
