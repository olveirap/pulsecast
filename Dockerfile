FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY pyproject.toml LICENSE README.md ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir $(python -c "import tomllib; deps=tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']; print(' '.join(deps))")

COPY pulsecast/ ./pulsecast/
COPY scripts/ ./scripts/
RUN pip install --no-cache-dir --no-deps .

ENV PYTHONUNBUFFERED=1

EXPOSE 8000
