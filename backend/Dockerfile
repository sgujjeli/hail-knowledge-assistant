# ── Stage 1: build ────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# System deps for PyMuPDF and Azure Speech SDK
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Runtime system deps (Azure Speech SDK needs these)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    ca-certificates \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app source (exclude .venv, .env, __pycache__)
COPY . .

# Azure App Service sets PORT env var; default to 8000
ENV PORT=8000

EXPOSE 8000

# Use shell form so $PORT is expanded at runtime
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
