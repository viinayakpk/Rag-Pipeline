from fastapi.testclient import TestClient

from app.api import document


def test_upload_document_rejects_invalid_metadata_json(
    client: TestClient,
    monkeypatch,
) -> None:
    monkeypatch.setattr(document, "_extract_pdf_text", lambda _content: "Example PDF text")

    response = client.post(
        "/document",
        files={"file": ("example.pdf", b"%PDF-1.4", "application/pdf")},
        data={"metadata": "{not-json"},
    )

    assert response.status_code == 400
    assert "metadata must be valid JSON" in response.json()["detail"]
