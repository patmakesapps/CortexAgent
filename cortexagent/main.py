from fastapi import FastAPI, Header, HTTPException

from cortexagent.config import settings
from cortexagent.models import (
    AgentChatRequest,
    AgentChatResponse,
    GoogleConnectRequest,
    GoogleConnectResponse,
)
from cortexagent.services import (
    AgentOrchestrator,
    ConnectedAccountsRepository,
    CortexLTMClient,
    GoogleOAuthService,
    resolve_user_id_from_authorization,
)
from cortexagent.tools import ToolRegistry, WebSearchTool

app = FastAPI(title="CortexAgent", version="0.1.0")


def _build_orchestrator() -> AgentOrchestrator:
    registry = ToolRegistry()
    if settings.web_search_enabled:
        registry.register(WebSearchTool())

    ltm_client = CortexLTMClient(
        base_url=settings.cortexltm_api_base_url,
        api_key=settings.cortexltm_api_key,
    )
    return AgentOrchestrator(ltm_client=ltm_client, tool_registry=registry)


orchestrator = _build_orchestrator()
connected_accounts_repo = ConnectedAccountsRepository(
    supabase_url=settings.supabase_url,
    supabase_service_role_key=settings.supabase_service_role_key,
    table=settings.connected_accounts_table,
    timeout_seconds=settings.connected_accounts_timeout_seconds,
)
google_oauth = GoogleOAuthService(
    client_id=settings.google_client_id,
    client_secret=settings.google_client_secret,
    redirect_uri=settings.google_redirect_uri,
    timeout_seconds=settings.google_oauth_timeout_seconds,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/agent/threads/{thread_id}/chat", response_model=AgentChatResponse)
def agent_chat(
    thread_id: str,
    payload: AgentChatRequest,
    authorization: str | None = Header(default=None),
) -> AgentChatResponse:
    try:
        result = orchestrator.handle_chat(
            thread_id=thread_id,
            text=payload.text,
            short_term_limit=payload.short_term_limit,
            authorization=authorization,
        )
        return AgentChatResponse(
            thread_id=thread_id,
            response=result.response,
            decision=result.decision,
            sources=result.sources,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected agent error: {exc}")


@app.post(
    "/v1/agent/integrations/google/connect",
    response_model=GoogleConnectResponse,
)
def connect_google_integration(
    payload: GoogleConnectRequest,
    authorization: str | None = Header(default=None),
) -> GoogleConnectResponse:
    try:
        user_id = resolve_user_id_from_authorization(
            authorization=authorization,
            supabase_url=settings.supabase_url,
            supabase_anon_key=settings.supabase_anon_key,
            timeout_seconds=5,
        )
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    try:
        account = google_oauth.connect_account(
            repo=connected_accounts_repo,
            user_id=user_id,
            code=payload.code,
            code_verifier=payload.code_verifier,
        )
        scopes = [value for value in (account.scope or "").split(" ") if value.strip()]
        return GoogleConnectResponse(
            provider=account.provider,
            provider_account_id=account.provider_account_id,
            user_id=account.user_id,
            status=account.status,
            scopes=scopes,
            expires_at=account.expires_at.isoformat() if account.expires_at else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected google connect error: {exc}")
