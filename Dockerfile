# Use a small official Python image
FROM python:3.12-slim

# Avoid warnings, set a working dir
WORKDIR /app

# Install system deps needed for many Python packages (optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt ./

# Install Python deps (fail early if requirements.txt bad)
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Railway provides $PORT env var at runtime; fall back to 8080
ENV PORT=8080

# Default command for production: use gunicorn and bind to $PORT
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:$PORT", "--workers", "1", "--threads", "4"]

# Entrypoint script (will be provided below)
ENTRYPOINT ["./docker-entrypoint.sh"]
