FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install
COPY requirements_deploy.txt .
RUN pip install --no-cache-dir -r requirements_deploy.txt

# Copy application files
COPY . .

# Create required directories
RUN mkdir -p uploads outputs demo_outputs sample_outputs checkpoints

# Expose FastAPI port
EXPOSE 8000

# Start server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
