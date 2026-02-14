import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency in constrained envs

    def load_dotenv(*_args, **_kwargs) -> bool:
        return False


load_dotenv(override=False)


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
    web_search_retries: int
    groq_api_key: str | None
    router_llm_enabled: bool
    router_llm_model: str
    router_llm_timeout_seconds: int
    supabase_url: str | None
    supabase_anon_key: str | None
    supabase_service_role_key: str | None
    connected_accounts_table: str
    connected_accounts_timeout_seconds: int
    google_client_id: str | None
    google_client_secret: str | None
    google_redirect_uri: str | None
    google_oauth_timeout_seconds: int


def load_settings() -> Settings:
    router_model = (
        os.getenv("AGENT_ROUTER_LLM_MODEL")
        or os.getenv("GROQ_ROUTER_MODEL")
        or os.getenv("GROQ_CHAT_MODEL")
        or "llama-3.1-8b-instant"
    )
    return Settings(
        cortexltm_api_base_url=os.getenv(
            "CORTEXLTM_API_BASE_URL", "http://127.0.0.1:8000"
        ),
        cortexltm_api_key=(os.getenv("CORTEXLTM_API_KEY") or None),
        agent_tools_enabled=_as_bool(os.getenv("AGENT_TOOLS_ENABLED"), True),
        web_search_enabled=_as_bool(os.getenv("WEB_SEARCH_ENABLED"), True),
        web_search_provider=os.getenv("WEB_SEARCH_PROVIDER", "duckduckgo,bing"),
        brave_search_api_key=(os.getenv("BRAVE_SEARCH_API_KEY") or None),
        web_search_timeout_seconds=_as_int(os.getenv("WEB_SEARCH_TIMEOUT_SECONDS"), 8),
        web_search_max_results=_as_int(os.getenv("WEB_SEARCH_MAX_RESULTS"), 5),
        web_search_retries=_as_int(os.getenv("WEB_SEARCH_RETRIES"), 2),
        groq_api_key=(os.getenv("GROQ_API_KEY") or None),
        router_llm_enabled=_as_bool(os.getenv("AGENT_ROUTER_LLM_ENABLED"), True),
        router_llm_model=router_model,
        router_llm_timeout_seconds=_as_int(
            os.getenv("AGENT_ROUTER_LLM_TIMEOUT_SECONDS"), 6
        ),
        supabase_url=(os.getenv("SUPABASE_URL") or None),
        supabase_anon_key=(os.getenv("SUPABASE_ANON_KEY") or None),
        supabase_service_role_key=(os.getenv("SUPABASE_SERVICE_ROLE_KEY") or None),
        connected_accounts_table=os.getenv(
            "CONNECTED_ACCOUNTS_TABLE", "ltm_connected_accounts"
        ),
        connected_accounts_timeout_seconds=_as_int(
            os.getenv("CONNECTED_ACCOUNTS_TIMEOUT_SECONDS"), 8
        ),
        google_client_id=(os.getenv("GOOGLE_CLIENT_ID") or None),
        google_client_secret=(os.getenv("GOOGLE_CLIENT_SECRET") or None),
        google_redirect_uri=(os.getenv("GOOGLE_REDIRECT_URI") or None),
        google_oauth_timeout_seconds=_as_int(
            os.getenv("GOOGLE_OAUTH_TIMEOUT_SECONDS"), 8
        ),
    )


settings = load_settings()
