import uuid

import pytest

from app.db.models import Chunk
from app.services import ingestion


class FakeScalarResult:
    def __init__(self, value):
        self._value = value

    def scalar_one(self):
        return self._value


class RecordingSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict | None]] = []

    async def execute(self, query, params=None):
        sql = str(query)
        self.calls.append((sql, params))
        if "INSERT INTO entities" in sql:
            return FakeScalarResult(params["id"])
        return FakeScalarResult(None)


@pytest.mark.asyncio
async def test_store_entities_writes_document_id_and_explicit_uuid(monkeypatch) -> None:
    monkeypatch.setattr(
        ingestion,
        "extract_entities",
        lambda _text, _language: [
            {"name": "European Commission", "normalized_name": "european commission", "entity_type": "ORG"}
        ],
    )

    document_id = uuid.uuid4()
    chunk = Chunk(id=uuid.uuid4(), document_id=document_id, chunk_index=0, text="hello")
    db = RecordingSession()

    await ingestion._store_entities(db, [chunk], ["hello"], "en")

    entity_sql, entity_params = db.calls[0]
    link_sql, link_params = db.calls[1]

    assert "INSERT INTO entities" in entity_sql
    assert isinstance(entity_params["id"], uuid.UUID)
    assert "INSERT INTO chunk_entities" in link_sql
    assert link_params["document_id"] == document_id
