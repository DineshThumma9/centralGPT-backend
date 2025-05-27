# === Base Builder Stage ===
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libffi-dev build-essential curl \
 && rm -rf /var/lib/apt/lists/*

# Install pipx + uv for faster Python dep management
RUN pip install --no-cache-dir pipx && pipx install uv
ENV PATH="/root/.local/bin:$PATH"

# Copy and install dependencies early to cache them
COPY pyproject.toml ./
RUN uv pip compile pyproject.toml -o requirements.txt

# Install Python dependencies from compiled requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source code (adjust path as needed)
COPY . .

# Expose port (if your app runs on a specific port, e.g., 8000)
EXPOSE 8000

# Run your app (replace with your actual start command)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
