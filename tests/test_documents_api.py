from fastapi.testclient import TestClient


def test_get_document_rejects_invalid_uuid(client: TestClient) -> None:
    response = client.get("/documents/not-a-uuid")

    assert response.status_code == 422
