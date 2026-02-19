from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import FastAPI, Header, HTTPException

from cortexagent.config import settings
from cortexagent.models import (
    AgentChatRequest,
    AgentChatResponse,
    GoogleConnectRequest,
    GoogleConnectResponse,
    GoogleConnectionStatusResponse,
)
from cortexagent.services.connected_accounts_repo import (
    ConnectedAccount,
    ConnectedAccountsRepository,
)
from cortexagent.services.cortexltm_client import CortexLtmClient
from cortexagent.services.executor import ToolExecutor
from cortexagent.services.google_oauth import GoogleOAuthService
from cortexagent.services.llm_client import OpenAICompatibleClient, OpenAICompatibleConfig
from cortexagent.services.orchestrator import AgentOrchestrator
from cortexagent.services.planner import LlmPlanner
from cortexagent.services.supabase_auth import resolve_user_id_from_authorization
from cortexagent.tools import GoogleCalendarTool, GoogleDriveTool, GoogleGmailTool, ToolRegistry

app = FastAPI(title="CortexAgent API", version="0.2.0")


def _build_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        tool=GoogleCalendarTool(),
        label="Google Calendar",
        description="Read upcoming events or create events in the connected Google Calendar.",
        schema={
            "type": "object",
            "required": [],
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "Operation: read or create.",
                },
                "event_text": {
                    "type": "string",
                    "description": "Event text for quick-add create operations.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum events to return for read operation.",
                },
            },
        },
    )
    registry.register(
        tool=GoogleDriveTool(),
        label="Google Drive",
        description="List or search files in the connected Google Drive account.",
        schema={
            "type": "object",
            "required": [],
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "Operation: search or list_recent.",
                },
                "query": {
                    "type": "string",
                    "description": "Drive search query string when operation=search.",
                },
            },
        },
    )
    registry.register(
        tool=GoogleGmailTool(),
        label="Gmail",
        description="List, read, draft, or send Gmail messages for the connected account.",
        schema={
            "type": "object",
            "required": [],
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "Operation: read, read_thread, draft_reply, draft_new, send.",
                },
                "thread_id": {
                    "type": "string",
                    "description": "Thread id for read_thread or draft_reply.",
                },
                "query": {
                    "type": "string",
                    "description": "Optional inbox query for read/list operations.",
                },
                "body": {
                    "type": "string",
                    "description": "Body text for drafting replies or new drafts.",
                },
                "to": {
                    "type": "string",
                    "description": "Recipient address for draft_new.",
                },
                "subject": {
                    "type": "string",
                    "description": "Subject line for draft_new.",
                },
                "draft_id": {
                    "type": "string",
                    "description": "Draft id for send operation.",
                },
            },
        },
    )
    return registry


def _build_synthesis_llm() -> OpenAICompatibleClient | None:
    if not settings.synthesis_llm_enabled:
        return None
    key = (settings.synthesis_llm_api_key or "").strip()
    if not key:
        return None
    return OpenAICompatibleClient(
        OpenAICompatibleConfig(
            provider=settings.synthesis_llm_provider,
            model=settings.synthesis_llm_model,
            api_key=key,
            api_base_url=settings.synthesis_llm_api_base_url,
            timeout_seconds=settings.synthesis_llm_timeout_seconds,
        )
    )


def _build_orchestrator(tool_registry: ToolRegistry) -> AgentOrchestrator | None:
    planner_key = (settings.planner_llm_api_key or "").strip()
    if not planner_key:
        return None
    planner_llm = OpenAICompatibleClient(
        OpenAICompatibleConfig(
            provider=settings.planner_llm_provider,
            model=settings.planner_llm_model,
            api_key=planner_key,
            api_base_url=settings.planner_llm_api_base_url,
            timeout_seconds=settings.planner_llm_timeout_seconds,
        )
    )
    return AgentOrchestrator(
        planner=LlmPlanner(llm=planner_llm, max_steps=settings.planner_llm_max_steps),
        executor=ToolExecutor(tool_registry=tool_registry),
        cortexltm=CortexLtmClient(
            base_url=settings.cortexltm_api_base_url,
            api_key=settings.cortexltm_api_key,
        ),
        synthesis_llm=_build_synthesis_llm(),
        planner_context_messages=settings.planner_context_messages,
    )


tool_registry = _build_tool_registry()
connected_accounts = ConnectedAccountsRepository(
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
orchestrator = _build_orchestrator(tool_registry)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/agent/threads/{thread_id}/chat", response_model=AgentChatResponse)
def chat_route(
    thread_id: str,
    payload: AgentChatRequest,
    authorization: str | None = Header(default=None),
) -> AgentChatResponse:
    if orchestrator is None:
        raise HTTPException(
            status_code=503,
            detail="Planner LLM key missing. Set AGENT_PLANNER_LLM_API_KEY or GROQ_API_KEY.",
        )
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Message text is required.")

    tool_meta = _resolve_google_tool_meta(authorization=authorization)
    try:
        result = orchestrator.chat(
            thread_id=thread_id,
            user_text=text,
            short_term_limit=payload.short_term_limit,
            authorization=authorization,
            planner_tool_registry_prompt=tool_registry.render_for_prompt(),
            tool_meta=tool_meta,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result.response


@app.post("/v1/agent/integrations/google/connect", response_model=GoogleConnectResponse)
def google_connect(
    payload: GoogleConnectRequest,
    authorization: str | None = Header(default=None),
) -> GoogleConnectResponse:
    user_id = _resolve_user_id(authorization)
    if not connected_accounts.is_configured():
        raise HTTPException(
            status_code=503,
            detail=(
                "Connected accounts repository is not configured. "
                "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY."
            ),
        )
    if not google_oauth.is_configured():
        raise HTTPException(
            status_code=503,
            detail=(
                "Google OAuth is not configured. "
                "Set GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, and GOOGLE_REDIRECT_URI."
            ),
        )

    try:
        account = google_oauth.connect_account(
            repo=connected_accounts,
            user_id=user_id,
            code=payload.code,
            code_verifier=payload.code_verifier,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    scopes = _split_scopes(account.scope)
    return GoogleConnectResponse(
        provider="google",
        provider_account_id=account.provider_account_id,
        user_id=account.user_id,
        status=account.status,
        scopes=scopes,
        expires_at=account.expires_at.isoformat() if account.expires_at else None,
    )


@app.get("/v1/agent/integrations/google/status", response_model=GoogleConnectionStatusResponse)
def google_status(
    authorization: str | None = Header(default=None),
) -> GoogleConnectionStatusResponse:
    user_id = _resolve_user_id(authorization)
    if not connected_accounts.is_configured():
        return GoogleConnectionStatusResponse(connected=False, scopes=[])
    account = connected_accounts.get_active_account(user_id=user_id, provider="google")
    if not account:
        return GoogleConnectionStatusResponse(connected=False, scopes=[])
    return GoogleConnectionStatusResponse(
        connected=True,
        scopes=_split_scopes(account.scope),
    )


@app.post("/v1/agent/integrations/google/disconnect")
def google_disconnect(
    authorization: str | None = Header(default=None),
) -> dict[str, object]:
    user_id = _resolve_user_id(authorization)
    if not connected_accounts.is_configured():
        return {"provider": "google", "disconnected": False}
    disconnected = connected_accounts.disconnect_provider(user_id=user_id, provider="google")
    return {"provider": "google", "disconnected": disconnected}


def _resolve_user_id(authorization: str | None) -> str:
    try:
        return resolve_user_id_from_authorization(
            authorization=authorization,
            supabase_url=settings.supabase_url,
            supabase_anon_key=settings.supabase_anon_key,
            timeout_seconds=5,
        )
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _resolve_google_tool_meta(authorization: str | None) -> dict[str, object]:
    if not connected_accounts.is_configured():
        return {}
    try:
        user_id = _resolve_user_id(authorization)
    except HTTPException:
        return {}
    token = connected_accounts.resolve_provider_token(user_id=user_id, provider="google")
    if token is None:
        return {}
    try:
        access_token = token.access_token
        if not access_token and token.refresh_token and google_oauth.is_configured():
            refreshed = google_oauth.refresh_access_token(token.refresh_token)
            expires_at = None
            if isinstance(refreshed.expires_in, int) and refreshed.expires_in > 0:
                expires_at = datetime.now(timezone.utc) + timedelta(seconds=refreshed.expires_in)
            account = connected_accounts.upsert_active_account(
                payload=_account_refresh_payload(
                    account=token.account,
                    access_token=refreshed.access_token,
                    refresh_token=refreshed.refresh_token or token.refresh_token,
                    token_type=refreshed.token_type,
                    scope=refreshed.scope or token.scope,
                    expires_at=expires_at,
                )
            )
            access_token = account.access_token
    except Exception:
        return {}
    out: dict[str, object] = {}
    if access_token:
        out["access_token"] = access_token
    return out


def _account_refresh_payload(
    *,
    account: ConnectedAccount,
    access_token: str,
    refresh_token: str | None,
    token_type: str | None,
    scope: str | None,
    expires_at: datetime | None,
) -> Any:
    from cortexagent.services.connected_accounts_repo import ConnectedAccountUpsert

    return ConnectedAccountUpsert(
        user_id=account.user_id,
        provider=account.provider,
        provider_account_id=account.provider_account_id,
        access_token=access_token,
        refresh_token=refresh_token,
        token_type=token_type,
        scope=scope,
        expires_at=expires_at,
        status="active",
        meta=account.meta,
    )


def _split_scopes(scope_text: str | None) -> list[str]:
    raw = (scope_text or "").strip()
    if not raw:
        return []
    return [token for token in raw.replace(",", " ").split(" ") if token]
