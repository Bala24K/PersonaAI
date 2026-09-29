import sys
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "2.0.0"
    assert "character" in data


def test_readiness_endpoint():
    response = client.get("/ready")
    assert response.status_code in [200, 503]
    data = response.json()
    assert "checks" in data
    assert "postgres" in data["checks"]
    assert "mongodb" in data["checks"]
    assert "redis" in data["checks"]
    assert "rabbitmq" in data["checks"]
    assert "cpp_analyzer" in data["checks"]
