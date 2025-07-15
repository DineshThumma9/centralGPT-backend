FROM python:3.10-slim AS builder

WORKDIR /app

# --- System dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# --- Python dependencies ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy app code ---
COPY . .

# --- Preload embedding models using huggingface_hub only ---
# Make model directories
RUN mkdir -p /models/bge /models/jina-code

# Preload BGE model files
RUN python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download('BAAI/bge-small-en-v1.5', 'model.safetensors', local_dir='/models/bge', local_dir_use_symlinks=False); \
hf_hub_download('BAAI/bge-small-en-v1.5', 'config.json', local_dir='/models/bge'); \
hf_hub_download('BAAI/bge-small-en-v1.5', 'tokenizer.json', local_dir='/models/bge') \
"

# Preload Jina Code model files
RUN python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download('jinaai/jina-embeddings-v2-base-code', 'model.safetensors', local_dir='/models/jina-code', local_dir_use_symlinks=False); \
hf_hub_download('jinaai/jina-embeddings-v2-base-code', 'config.json', local_dir='/models/jina-code'); \
hf_hub_download('jinaai/jina-embeddings-v2-base-code', 'tokenizer.json', local_dir='/models/jina-code') \
"

# --- Set env and expose port ---
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
