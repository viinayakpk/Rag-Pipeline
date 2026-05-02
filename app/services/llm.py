from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.errors import ServiceDependencyError


def get_active_model_name() -> str:
    settings = get_settings()
    if settings.llm_provider == "mock":
        return "mock"
    if settings.llm_provider == "ollama":
        return settings.ollama_model
    if settings.llm_provider == "openrouter":
        return settings.openrouter_model
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider!r}")


@lru_cache(maxsize=1)
def _get_openrouter_client():
    try:
        from openai import AsyncOpenAI
    except ModuleNotFoundError as exc:
        raise ServiceDependencyError(
            "openai is required for the OpenRouter adapter. "
            "Install project dependencies before using this provider."
        ) from exc

    settings = get_settings()
    return AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )


async def generate_text_response(prompt: str) -> str:
    settings = get_settings()

    if settings.llm_provider == "mock":
        return f"[mock] This is a placeholder answer. Prompt preview: {prompt[:120]!r}"

    if settings.llm_provider == "openrouter":
        client = _get_openrouter_client()
        try:
            response = await client.chat.completions.create(
                model=settings.openrouter_model,
                messages=[{"role": "user", "content": prompt}],
                timeout=60.0,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise ServiceDependencyError(
                f"OpenRouter request failed ({exc.__class__.__name__}): {exc}"
            ) from exc

    if settings.llm_provider == "ollama":
        try:
            import httpx
        except ModuleNotFoundError as exc:
            raise ServiceDependencyError(
                "httpx is required for the Ollama adapter. "
                "Install project dependencies before using this provider."
            ) from exc
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/api/generate",
                json={
                    "model": settings.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                },
            )
            response.raise_for_status()
            return response.json().get("response", "")

    raise ValueError(f"Unknown LLM provider: {settings.llm_provider!r}")
