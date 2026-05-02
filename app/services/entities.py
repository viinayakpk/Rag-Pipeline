from __future__ import annotations

from functools import lru_cache

from app.errors import ServiceDependencyError

# Norwegian has no dedicated spaCy model; xx_ent_wiki_sm is the multilingual fallback.
_MODEL_MAP: dict[str, str] = {
    "en": "en_core_web_sm",
    "de": "de_core_news_sm",
    "fr": "fr_core_news_sm",
    "sv": "sv_core_news_sm",
    "no": "xx_ent_wiki_sm",
}
_DEFAULT_MODEL = "xx_ent_wiki_sm"

_KEPT_TYPES = {"PERSON", "ORG", "GPE", "LOC", "NORP", "EVENT"}
_MAX_CHARS = 8000


@lru_cache(maxsize=8)
def _load_nlp(model_name: str):
    try:
        import spacy
    except ModuleNotFoundError as exc:
        raise ServiceDependencyError(
            "spaCy is required for entity extraction. "
            "Install project dependencies before calling ingestion."
        ) from exc
    return spacy.load(model_name)


def extract_entities(text: str, language: str) -> list[dict[str, str]]:
    model_name = _MODEL_MAP.get(language, _DEFAULT_MODEL)
    nlp = _load_nlp(model_name)
    doc = nlp(text[:_MAX_CHARS])
    seen: set[tuple[str, str]] = set()
    results: list[dict[str, str]] = []
    for ent in doc.ents:
        if ent.label_ not in _KEPT_TYPES or not ent.text.strip():
            continue
        key = (ent.text.lower().strip(), ent.label_)
        if key not in seen:
            seen.add(key)
            results.append({
                "name": ent.text,
                "normalized_name": ent.text.lower().strip(),
                "entity_type": ent.label_,
            })
    return results
