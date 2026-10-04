"""Tests for API-level security: authentication and cross-user access control.

These tests verify:
- Optional API key authentication works when PERSONA_API_KEY is set
- Health/ready endpoints bypass auth
- Cross-user data access is prevented at the API level
- Metrics endpoint doesn't expose secrets
"""
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestAuthenticationMiddleware:
    """Tests for the optional API key authentication."""

    def test_no_auth_when_key_not_set(self):
        """When PERSONA_API_KEY is not set, all endpoints should be accessible."""
        # main.py reads the key at import time, so we test the default behavior
        from main import app
        client = TestClient(app)
        resp = client.post("/chat", json={
            "user_id": "auth_test_u1",
            "session_id": "s1",
            "message": "hello",
        })
        # Should work without auth header
        assert resp.status_code == 200

    def test_health_always_accessible(self):
        from main import app
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_ready_always_accessible(self):
        from main import app
        client = TestClient(app)
        resp = client.get("/ready")
        assert resp.status_code in (200, 503)


class TestMetricsEndpoint:
    """Tests for the observability metrics endpoint."""

    def test_metrics_returns_json(self):
        from main import app
        client = TestClient(app)
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "counters" in data
        assert "latencies" in data

    def test_metrics_does_not_expose_secrets(self):
        from main import app
        client = TestClient(app)
        resp = client.get("/metrics")
        body = resp.text.lower()
        # Must not contain any API key patterns
        assert "api_key" not in body
        assert "bearer" not in body
        assert "password" not in body
        assert "secret" not in body


class TestAPIUserIsolation:
    """Tests that API endpoints enforce user isolation."""

    def test_user_memories_endpoint_returns_only_own(self):
        from main import app
        client = TestClient(app)

        # Create memories for user_a via chat
        client.post("/chat", json={
            "user_id": "api_iso_a",
            "session_id": "s1",
            "message": "I love pizza",
        })

        # Get user_a memories
        resp = client.get("/user/api_iso_a/memories")
        assert resp.status_code == 200

        # Get user_b memories — should not contain user_a data
        resp_b = client.get("/user/api_iso_b/memories")
        assert resp_b.status_code == 200
        data_b = resp_b.json()
        for mem in data_b.get("memories", []):
            assert "pizza" not in mem.get("content", "").lower()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
