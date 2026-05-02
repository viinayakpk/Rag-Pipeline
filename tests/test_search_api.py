from fastapi.testclient import TestClient


def test_search_rejects_blank_query(client: TestClient) -> None:
    response = client.get("/search", params={"q": "   "})

    assert response.status_code == 400
    assert response.json()["detail"] == "Search query must not be empty"
