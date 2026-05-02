from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_host: str = "0.0.0.0"
    app_port: int = 8000

    postgres_db: str = "ragkb"
    postgres_user: str = "raguser"
    postgres_password: str = "ragpass"
    database_url: str = Field(
        default="postgresql+asyncpg://raguser:ragpass@db:5432/ragkb"
    )

    embedding_model: str = "intfloat/multilingual-e5-base"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    llm_provider: Literal["mock", "ollama", "openrouter"] = "mock"
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "llama3.2:3b"
    openrouter_api_key: str = ""
    openrouter_model: str = "meta-llama/llama-3.1-8b-instruct:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    default_search_limit: int = 10
    hybrid_candidate_count: int = 30
    chat_context_top_k: int = 5

    api_key: str = ""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def clear_settings_cache() -> None:
    get_settings.cache_clear()

