FROM python:3.10-slim

WORKDIR /app

# Install system dependencies: ffmpeg for video processing, build tools for
# native Python extensions (insightface / onnxruntime).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies.
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application and worker code.
COPY backend/app/ ./app/
COPY worker/ ./worker/

# The data/ directory is NOT baked into the image — it is bind-mounted at
# runtime (./data:/app/data) so api and worker share the same SQLite DB and
# per-job files.

# Configure Python logging to stdout at INFO level for both api and worker.
ENV PYTHONUNBUFFERED=1

# Default command: run FastAPI app.  Override in docker-compose for the worker.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--log-level", "info"]
