from app.config import Settings


def test_settings_can_be_instantiated_for_mock_mode() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql+asyncpg://raguser:ragpass@db:5432/ragkb",
        llm_provider="mock",
    )

    assert settings.database_url.endswith("/ragkb")
    assert settings.llm_provider == "mock"
    assert settings.ollama_base_url == "http://ollama:11434"

