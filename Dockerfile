FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application
COPY . .

# --- Preload models using huggingface_hub only ---
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-small-en-v1.5')"
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('jinaai/jina-embeddings-v2-base-code')"

# Set environment
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
