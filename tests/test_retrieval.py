import json

from app.schemas.search import SearchFilters
from app.services import retrieval


def test_build_filter_sql_uses_jsonb_cast_for_topic() -> None:
    sql, params = retrieval._build_filter_sql(SearchFilters(topic="energy", language="en"))

    assert "c.topics @> CAST(:f_topic AS JSONB)" in sql
    assert params["f_topic"] == json.dumps(["energy"])
    assert params["f_language"] == "en"


def test_get_fts_config_prefers_language_specific_config() -> None:
    assert retrieval._get_fts_configs("en") == ("english",)
    assert retrieval._get_fts_configs("sv") == ("swedish",)
    assert retrieval._get_fts_configs(None) == (
        "english",
        "german",
        "french",
        "swedish",
        "norwegian",
        "simple",
    )
