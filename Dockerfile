# Use a small official Python image
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system deps needed for common wheels (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements if present; otherwise we'll fall back and install core deps
COPY requirements.txt .

RUN if [ -f requirements.txt ]; then \
      pip install --no-cache-dir -r requirements.txt ; \
    else \
      pip install --no-cache-dir flask gunicorn python-dotenv requests ; \
    fi

# Copy the app code
COPY . .

# Create an unprivileged user and use it
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app

USER appuser

# Expose app port
EXPOSE 5000

# Entrypoint script (will be provided below)
ENTRYPOINT ["./docker-entrypoint.sh"]
