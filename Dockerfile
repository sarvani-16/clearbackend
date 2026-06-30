FROM python:3.10-slim

# Install system dependencies for OpenCV, PostgreSQL, and general builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libpq-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements_deploy.txt .
RUN pip install --no-cache-dir -r requirements_deploy.txt

# Copy application files
COPY . .

# Create required directories
RUN mkdir -p uploads outputs demo_outputs sample_outputs checkpoints

# Expose default port
EXPOSE 8000

# Start server using shell form to expand $PORT dynamically on Railway (falls back to 8000)
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
