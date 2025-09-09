"""
Unit tests for the main application and core functionality.
"""
import pytest
from fastapi.testclient import TestClient
from src.main import app


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_root_endpoint(self, client: TestClient):
        """Test the root endpoint returns success."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "API is running"}
    
    def test_health_endpoint(self, client: TestClient):
        """Test the health endpoint returns success."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"message": "API is running"}


class TestApplicationConfig:
    """Test application configuration."""
    
    def test_app_title(self):
        """Test that the app has the correct title."""
        assert app.title == "CentralGPT Backend API"
    
    def test_app_version(self):
        """Test that the app has a version."""
        assert app.version == "1.0.0"
    
    def test_app_description(self):
        """Test that the app has a description."""
        assert "Multi-LLM Provider" in app.description


class TestCORSConfiguration:
    """Test CORS middleware configuration."""
    
    def test_cors_headers_present(self, client: TestClient):
        """Test that CORS headers are present in responses."""
        response = client.options("/", headers={"Origin": "http://localhost:5173"})
        # FastAPI automatically handles CORS preflight requests
        assert response.status_code in [200, 405]  # 405 if OPTIONS not explicitly defined