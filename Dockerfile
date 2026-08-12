# ---------------------------------------------------------------------------
# Phishing Website Detection System - Backend + Frontend container
# ---------------------------------------------------------------------------
# Builds a self-contained image that serves the FastAPI backend AND the
# static React dashboard (mounted at /app) on port 8000.
#
# Build:  docker build -t phishing-detector .
# Run:    docker run -p 8000:8000 phishing-detector
# Then open http://localhost:8000/app in your browser.
# ---------------------------------------------------------------------------

FROM python:3.11-slim

# --- Environment hardening / sane Python defaults ---
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# --- System build dependencies ---
# libgomp1  -> required by xgboost / lightgbm OpenMP runtime
# build-essential -> compile any wheels without prebuilt binaries
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Install Python dependencies first (better layer caching) ---
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# --- Copy the full project (code, trained models, frontend, results) ---
COPY . .

# --- Persist prediction history / logs on a mounted volume if desired ---
RUN mkdir -p data logs
VOLUME ["/app/data", "/app/logs"]

EXPOSE 8000

# --- Container healthcheck hits the API /health endpoint ---
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# --- Launch the API (also serves the dashboard at /app) ---
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
