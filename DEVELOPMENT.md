# Development Guide

## Prerequisites

- Python 3.10+
- PostgreSQL
- Redis
- Qdrant (vector database)

## Development Setup

1. **Clone and setup environment:**
   ```bash
   git clone https://github.com/DineshThumma9/centralGPT-backend.git
   cd centralGPT-backend
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Set up pre-commit hooks:**
   ```bash
   pre-commit install
   ```

## Code Quality Tools

### Formatting and Linting
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/

# Security scan
bandit -r src/
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test categories
pytest -m unit
pytest -m integration
```

## Code Standards

- Use **Black** for code formatting (88 character line limit)
- Use **isort** for import sorting
- Follow **PEP 8** style guidelines
- Write **type hints** for all function parameters and return values
- Add **docstrings** for all public functions and classes
- Maintain **test coverage** above 80%

## Project Structure

```
src/
├── main.py              # FastAPI application entry point
├── models/              # Pydantic models and database schemas
│   └── schema.py
├── router/              # API route handlers
│   ├── auth.py          # Authentication routes
│   ├── messages.py      # Message handling routes
│   ├── rag.py          # RAG functionality routes
│   └── ...
├── service/             # Business logic layer
│   ├── auth_service.py  # Authentication logic
│   ├── rag_service.py   # RAG processing logic
│   └── ...
└── db/                  # Database connection and utilities
    ├── dbs.py           # Main database connection
    ├── redis_client.py  # Redis connection
    └── qdrant_client.py # Vector database connection

tests/
├── unit/                # Unit tests
├── integration/         # Integration tests
└── conftest.py         # Test fixtures and configuration
```

## API Documentation

When running the development server, API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Environment Variables

See `.env.example` for required environment variables. Key configurations:

- **Database**: PostgreSQL connection string
- **Redis**: Redis connection for caching and sessions
- **Qdrant**: Vector database for RAG functionality
- **API Keys**: Various LLM provider keys (Groq, OpenAI, etc.)
- **Security**: JWT secret key and token expiry settings

## Common Development Tasks

### Adding a New API Endpoint
1. Define request/response models in `src/models/schema.py`
2. Implement business logic in appropriate service module
3. Create route handler in relevant router module
4. Add router to main.py if new router
5. Write unit tests for the new functionality

### Adding a New LLM Provider
1. Add provider configuration to service layer
2. Update schema models for provider-specific settings
3. Implement provider-specific logic in chat service
4. Add provider to frontend configuration options
5. Update documentation

## Troubleshooting

### Common Issues

1. **Database connection errors**: Check DATABASE_URL in .env
2. **Redis connection errors**: Ensure Redis is running and URL is correct
3. **Import errors**: Make sure you're in the project root and dependencies are installed
4. **Permission errors**: Check file permissions and virtual environment activation

### Logging

Logs are written to:
- `logs/api.log` (structured API logs)
- `logs/app.log` (application logs)

Adjust log levels in the environment variables or main.py configuration.