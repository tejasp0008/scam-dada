"""The app boots, wires its providers and reports them."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_healthz_reports_the_environment(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "app_env": "test"}


def test_readyz_names_the_wired_providers(client: TestClient) -> None:
    """Readiness has to show which adapters are live, so a demo-day misconfiguration
    (mock TTS still selected) is one request away rather than a surprise on stage."""
    body = client.get("/readyz").json()
    assert body["providers"] == {"stt": "mock-stt", "tts": "mock-tts", "llm": "mock-llm"}
    assert set(body) == {"ready", "postgres", "redis", "providers"}


def test_readyz_is_unavailable_when_a_datastore_is_down(client: TestClient) -> None:
    """Whatever is reachable locally, the contract is: not ready implies 503."""
    response = client.get("/readyz")
    body = response.json()
    assert body["ready"] == (body["postgres"] and body["redis"])
    assert response.status_code == (200 if body["ready"] else 503)
