# req.Dockerfile
FROM python:3.12-slim

WORKDIR /deps

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

# Only copy requirements and install
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Freeze environment to reuse later
RUN pip freeze > installed.txt
