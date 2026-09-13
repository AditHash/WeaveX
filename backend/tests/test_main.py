"""Smoke test for the FastAPI app itself.

Uses FastAPI's TestClient (in-process ASGI calls, no real socket/process)
rather than spinning up uvicorn — faster, and avoids OS-level file locks
on a running server process.
"""

from fastapi.testclient import TestClient

from app.main import app


def test_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
