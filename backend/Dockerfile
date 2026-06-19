FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (including git for pip if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application and worker code
COPY backend/app/ ./app/
COPY worker/ ./worker/

# Default command: run FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
