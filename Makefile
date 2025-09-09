.PHONY: help install install-dev format lint test test-cov clean security run

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies
	pip install -r requirements.txt

install-dev:  ## Install development dependencies
	pip install -r requirements.txt -r requirements-dev.txt
	pre-commit install

format:  ## Format code with black and isort
	black src/ tests/
	isort src/ tests/

lint:  ## Run linting checks
	flake8 src/ tests/
	mypy src/

test:  ## Run tests
	pytest tests/

test-cov:  ## Run tests with coverage
	pytest --cov=src --cov-report=html --cov-report=term tests/

security:  ## Run security checks
	bandit -r src/
	safety check

clean:  ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ .pytest_cache/ build/ dist/

run:  ## Run the development server
	uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

check: format lint test security  ## Run all code quality checks

setup: install-dev  ## Complete development setup
	@echo "Development environment setup complete!"
	@echo "Next steps:"
	@echo "1. Copy .env.example to .env and configure your environment"
	@echo "2. Run 'make run' to start the development server"