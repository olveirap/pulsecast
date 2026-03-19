FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY pyproject.toml LICENSE README.md ./

RUN pip install --upgrade pip && pip install .


# Copy application source
COPY pulsecast/ ./pulsecast/

ENV PYTHONUNBUFFERED=1

EXPOSE 8000
