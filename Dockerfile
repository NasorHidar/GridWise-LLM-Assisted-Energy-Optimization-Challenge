# GridWise — minimal Docker image for the BUP CSE Fest 2026 preliminary.
#
# No environment variables, API keys, or secrets are baked in. The container
# only listens on 0.0.0.0:8000 and serves the FastAPI app.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies. PuLP ships with the bundled CBC solver, so
# no extra apt packages are required.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

# Bind 0.0.0.0 with a single worker (CBC is single-threaded; GIL does not
# matter here). The container exposes /health immediately on startup.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
