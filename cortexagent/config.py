import os
from dataclasses import dataclass


def _as_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(raw: str | None, default: int) -> int:
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    cortexltm_api_base_url: str
    cortexltm_api_key: str | None
    agent_tools_enabled: bool
    web_search_enabled: bool
    web_search_provider: str
    brave_search_api_key: str | None
    web_search_timeout_seconds: int
    web_search_max_results: int


def load_settings() -> Settings:
    return Settings(
        cortexltm_api_base_url=os.getenv("CORTEXLTM_API_BASE_URL", "http://127.0.0.1:8000"),
        cortexltm_api_key=(os.getenv("CORTEXLTM_API_KEY") or None),
        agent_tools_enabled=_as_bool(os.getenv("AGENT_TOOLS_ENABLED"), True),
        web_search_enabled=_as_bool(os.getenv("WEB_SEARCH_ENABLED"), True),
        web_search_provider=os.getenv("WEB_SEARCH_PROVIDER", "duckduckgo"),
        brave_search_api_key=(os.getenv("BRAVE_SEARCH_API_KEY") or None),
        web_search_timeout_seconds=_as_int(os.getenv("WEB_SEARCH_TIMEOUT_SECONDS"), 8),
        web_search_max_results=_as_int(os.getenv("WEB_SEARCH_MAX_RESULTS"), 5),
    )


settings = load_settings()
