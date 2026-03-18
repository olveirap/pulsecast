FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[dev]" 2>/dev/null || \
    pip install --no-cache-dir .

# Copy application source
COPY pulsecast/ ./pulsecast/

ENV PYTHONUNBUFFERED=1

EXPOSE 8000
