from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient

from app.config import clear_settings_cache
from app.db.session import get_db
from app.main import app


class FakeResult:
    def fetchone(self):
        return None

    def fetchall(self):
        return []


class FakeSession:
    async def execute(self, _query, _params=None):
        return FakeResult()

    def add(self, _obj): pass

    async def flush(self): pass

    async def commit(self): pass


@pytest.fixture()
def client(monkeypatch) -> TestClient:
    monkeypatch.setenv("API_KEY", "")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    clear_settings_cache()

    async def override_get_db() -> AsyncIterator[FakeSession]:
        yield FakeSession()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    clear_settings_cache()
