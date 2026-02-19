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
    planner_llm_api_key: str | None
    planner_llm_provider: str
    planner_llm_model: str
    planner_llm_api_base_url: str | None
    planner_llm_timeout_seconds: int
    planner_llm_max_steps: int
    planner_context_messages: int
    synthesis_llm_enabled: bool
    synthesis_llm_provider: str
    synthesis_llm_model: str
    synthesis_llm_api_base_url: str | None
    synthesis_llm_api_key: str | None
    synthesis_llm_timeout_seconds: int
    supabase_url: str | None
    supabase_anon_key: str | None
    supabase_service_role_key: str | None
    connected_accounts_table: str
    connected_accounts_timeout_seconds: int
    google_client_id: str | None
    google_client_secret: str | None
    google_redirect_uri: str | None
    google_oauth_timeout_seconds: int
    gmail_allowed_recipient_domains: str


def load_settings() -> Settings:
    planner_model = (
        os.getenv("AGENT_PLANNER_LLM_MODEL")
        or os.getenv("GROQ_CHAT_MODEL")
        or "llama-3.1-8b-instant"
    )
    planner_provider = os.getenv("AGENT_PLANNER_LLM_PROVIDER", "groq").strip().lower()
    planner_key = os.getenv("AGENT_PLANNER_LLM_API_KEY") or os.getenv("GROQ_API_KEY") or None
    return Settings(
        cortexltm_api_base_url=os.getenv(
            "CORTEXLTM_API_BASE_URL", "http://127.0.0.1:8000"
        ),
        cortexltm_api_key=(os.getenv("CORTEXLTM_API_KEY") or None),
        planner_llm_api_key=planner_key,
        planner_llm_provider=planner_provider,
        planner_llm_model=planner_model,
        planner_llm_api_base_url=(os.getenv("AGENT_PLANNER_LLM_API_BASE_URL") or None),
        planner_llm_timeout_seconds=_as_int(
            os.getenv("AGENT_PLANNER_LLM_TIMEOUT_SECONDS"), 8
        ),
        planner_llm_max_steps=max(1, min(8, _as_int(os.getenv("AGENT_PLANNER_MAX_STEPS"), 4))),
        planner_context_messages=max(
            4,
            min(24, _as_int(os.getenv("AGENT_PLANNER_CONTEXT_MESSAGES"), 10)),
        ),
        synthesis_llm_enabled=_as_bool(os.getenv("AGENT_SYNTHESIS_LLM_ENABLED"), True),
        synthesis_llm_provider=os.getenv(
            "AGENT_SYNTHESIS_LLM_PROVIDER", planner_provider
        ).strip().lower(),
        synthesis_llm_model=(
            os.getenv("AGENT_SYNTHESIS_LLM_MODEL")
            or planner_model
        ),
        synthesis_llm_api_base_url=(os.getenv("AGENT_SYNTHESIS_LLM_API_BASE_URL") or None),
        synthesis_llm_api_key=(
            os.getenv("AGENT_SYNTHESIS_LLM_API_KEY")
            or planner_key
        ),
        synthesis_llm_timeout_seconds=_as_int(
            os.getenv("AGENT_SYNTHESIS_LLM_TIMEOUT_SECONDS"), 10
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
        gmail_allowed_recipient_domains=os.getenv(
            "GMAIL_ALLOWED_RECIPIENT_DOMAINS", ""
        ),
    )


settings = load_settings()
