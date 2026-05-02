from fastapi.testclient import TestClient
import app.main as main_module


def test_health_endpoint_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "database": "ok",
        "llm_provider": "mock",
    }


def test_ready_endpoint_reports_ok_when_runtime_dependencies_are_available(
    client: TestClient,
    monkeypatch,
) -> None:
    monkeypatch.setattr(main_module, "_missing_runtime_dependencies", lambda: [])

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "database": "ok",
        "dependencies": "ok",
        "llm_provider": "mock",
    }
