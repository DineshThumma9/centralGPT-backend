# Dockerfile
FROM abc-deps as base

WORKDIR /app

# Copy only your application code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
