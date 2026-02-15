from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
import json
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib import request as urlrequest

from cortexagent.config import settings
from cortexagent.models import AgentDecision
from cortexagent.router import RouteDecision, decide_action
from cortexagent.services.connected_accounts_repo import (
    ConnectedAccountsRepository,
    ConnectedAccountUpsert,
)
from cortexagent.services.cortexltm_client import CortexLTMClient
from cortexagent.services.google_oauth import GoogleOAuthService
from cortexagent.services.supabase_auth import resolve_user_id_from_authorization
from cortexagent.services.verification import (
    assess_verification_profile,
    enforce_verification_policy,
)
from cortexagent.tools import ToolContext, ToolRegistry


@dataclass(frozen=True)
class OrchestratorResult:
    response: str
    decision: AgentDecision
    sources: list[dict[str, str]]
    tool_pipeline: list[dict[str, object]] | None = None


@dataclass(frozen=True)
class OrchestrationStep:
    action: str
    query: str


@dataclass(frozen=True)
class OrchestrationStepResult:
    action: str
    tool_name: str
    query: str
    assistant_text: str
    sources: list[dict[str, str]]
    success: bool
    reason: str
    capability_label: str
    extra_meta: dict[str, object] | None = None


class AgentOrchestrator:
    def __init__(
        self,
        ltm_client: CortexLTMClient,
        tool_registry: ToolRegistry,
        connected_accounts_repo: ConnectedAccountsRepository | None = None,
        google_oauth: GoogleOAuthService | None = None,
    ) -> None:
        self.ltm_client = ltm_client
        self.tool_registry = tool_registry
        self.connected_accounts_repo = connected_accounts_repo
        self.google_oauth = google_oauth
        self._pending_calendar_drafts: dict[str, str] = {}
        self._pending_gmail_send_requests: dict[str, str] = {}
        self._last_tool_action_by_thread: dict[str, str] = {}
        self._last_non_meta_user_text_by_thread: dict[str, str] = {}
        self._last_web_search_query_by_thread: dict[str, str] = {}
        self._last_pipeline_results_by_thread: dict[str, list[dict[str, object]]] = {}

    def handle_chat(
        self,
        thread_id: str,
        text: str,
        short_term_limit: int | None,
        authorization: str | None,
    ) -> OrchestratorResult:
        previous_non_meta_user_text = self._last_non_meta_user_text_by_thread.get(thread_id)
        is_meta_web_request = _is_meta_web_search_request(text)
        if not is_meta_web_request:
            self._last_non_meta_user_text_by_thread[thread_id] = text

        pending_calendar_draft = self._pending_calendar_drafts.get(thread_id)
        pending_gmail_send = self._pending_gmail_send_requests.get(thread_id)
        if pending_calendar_draft and _is_calendar_cancel_reply(text):
            self._pending_calendar_drafts.pop(thread_id, None)
            response = "Understood. I canceled this calendar draft and did not add anything."
            return OrchestratorResult(
                response=response,
                decision=AgentDecision(
                    action="google_calendar",
                    reason="calendar_draft_canceled",
                    confidence=1.0,
                ),
                sources=[],
            )
        if pending_calendar_draft and _is_calendar_pause_reply(text):
            response = (
                "Understood. I will hold this calendar draft for now. "
                "Reply with 'confirm' when you want me to add it."
            )
            return OrchestratorResult(
                response=response,
                decision=AgentDecision(
                    action="google_calendar",
                    reason="calendar_draft_paused",
                    confidence=1.0,
                ),
                sources=[],
            )
        if pending_gmail_send and _is_gmail_cancel_reply(text):
            self._pending_gmail_send_requests.pop(thread_id, None)
            response = "Understood. I canceled this Gmail send request and did not send anything."
            return OrchestratorResult(
                response=response,
                decision=AgentDecision(
                    action="google_gmail",
                    reason="gmail_send_canceled",
                    confidence=1.0,
                ),
                sources=[],
            )
        if (
            not pending_calendar_draft
            and not pending_gmail_send
            and _is_standalone_calendar_confirmation_reply(text)
        ):
            response = (
                "I don't have a pending calendar draft to confirm right now. "
                "Tell me the event details, and I'll draft it for your approval."
            )
            return OrchestratorResult(
                response=response,
                decision=AgentDecision(
                    action="google_calendar",
                    reason="calendar_confirmation_missing_draft",
                    confidence=1.0,
                ),
                sources=[],
            )

        if settings.agent_tools_enabled:
            pipeline_steps = _build_multi_step_plan(text)
            if pipeline_steps:
                return self._run_multi_step_pipeline(
                    thread_id=thread_id,
                    text=text,
                    authorization=authorization,
                    steps=pipeline_steps,
                )

        verification = assess_verification_profile(text)
        if pending_calendar_draft and _is_calendar_draft_edit_reply(text):
            route = RouteDecision(
                action="google_calendar",
                reason="calendar_confirmation_followup",
                confidence=0.99,
            )
        elif pending_calendar_draft:
            route = RouteDecision(
                action="google_calendar",
                reason="calendar_pending_default_confirm",
                confidence=0.99,
            )
        elif pending_gmail_send and _is_gmail_send_confirmation_reply(text):
            route = RouteDecision(
                action="google_gmail",
                reason="gmail_send_confirmation_followup",
                confidence=0.99,
            )
        elif (
            (_is_generic_web_followup_request(text) or is_meta_web_request)
            and (
                self._last_web_search_query_by_thread.get(thread_id)
                or previous_non_meta_user_text
            )
        ):
            route = RouteDecision(
                action="web_search",
                reason="web_search_followup",
                confidence=0.99,
            )
        else:
            route = decide_action(
                user_text=text,
                tools_enabled=settings.agent_tools_enabled,
                web_search_enabled=settings.web_search_enabled,
                prior_user_text=previous_non_meta_user_text,
                prior_tool_action=self._last_tool_action_by_thread.get(thread_id),
            )
            if route.action == "chat":
                sticky_route = _maybe_continue_recent_tool_route(
                    text=text,
                    previous_action=self._last_tool_action_by_thread.get(thread_id),
                )
                if sticky_route is not None:
                    route = sticky_route
        if (
            verification.requires_web_verification
            and settings.agent_tools_enabled
            and settings.web_search_enabled
            and route.action != "web_search"
            and _should_force_verification_web_search(text, verification.reasons)
        ):
            route = RouteDecision(
                action="web_search",
                reason=f"verification_override:{','.join(verification.reasons)}",
                confidence=max(route.confidence, 0.9),
            )
        if route.action in {"google_calendar", "google_gmail", "google_drive"}:
            self._last_tool_action_by_thread[thread_id] = route.action

        if route.action not in {"web_search", "google_calendar", "google_gmail", "google_drive"}:
            assistant_text = self.ltm_client.chat(
                thread_id=thread_id,
                text=text,
                short_term_limit=short_term_limit,
                authorization=authorization,
            )
            assistant_text = _polish_response_text(assistant_text)
            assistant_text = enforce_verification_policy(
                user_text=text,
                assistant_text=assistant_text,
                sources=[],
                profile=verification,
            )
            assistant_text = _prevent_unexecuted_action_claims(
                assistant_text=assistant_text,
                routed_action=route.action,
            )
            return OrchestratorResult(
                response=assistant_text,
                decision=AgentDecision(
                    action=route.action,
                    reason=route.reason,
                    confidence=route.confidence,
                ),
                sources=[],
            )

        if route.action == "google_calendar":
            tool = self.tool_registry.get("google_calendar")
            effective_calendar_text = text
            if pending_calendar_draft and not _is_calendar_draft_edit_reply(text):
                effective_calendar_text = _build_confirmed_calendar_request(
                    draft_text=pending_calendar_draft,
                    followup_text=text,
                )
            elif pending_calendar_draft and _is_calendar_draft_edit_reply(text):
                effective_calendar_text = _build_updated_calendar_draft_request(
                    draft_text=pending_calendar_draft,
                    followup_text=text,
                )
            try:
                access_token, token_meta = self._resolve_google_access_token(
                    authorization=authorization,
                    integration_label="Google Calendar",
                )
            except Exception as exc:
                assistant_text = str(exc)
                self._persist_nonfatal_tool_error_events(
                    thread_id=thread_id,
                    user_text=text,
                    assistant_text=assistant_text,
                    query=effective_calendar_text,
                    authorization=authorization,
                    decision_action="google_calendar",
                    tool_name="google_calendar",
                    capability_label="Google Calendar",
                )
                return OrchestratorResult(
                    response=assistant_text,
                    decision=AgentDecision(
                        action="google_calendar",
                        reason="google_calendar_auth_failed",
                        confidence=1.0,
                    ),
                    sources=[],
                )

            try:
                result = tool.run(
                    ToolContext(
                        thread_id=thread_id,
                        user_text=effective_calendar_text,
                        tool_meta={
                            "access_token": access_token,
                            "max_results": 8,
                        },
                    )
                )
            except Exception as exc:
                assistant_text = (
                    "I routed this request to Google Calendar, but the request failed. "
                    f"Error: {exc}"
                )
                self._persist_nonfatal_tool_error_events(
                    thread_id=thread_id,
                    user_text=text,
                    assistant_text=assistant_text,
                    query=effective_calendar_text,
                    authorization=authorization,
                    decision_action="google_calendar",
                    tool_name="google_calendar",
                    capability_label="Google Calendar",
                )
                return OrchestratorResult(
                    response=assistant_text,
                    decision=AgentDecision(
                        action="google_calendar",
                        reason="google_calendar_failed",
                        confidence=1.0,
                    ),
                    sources=[],
                )

            assistant_text, sources = _format_google_calendar_response(result.items)
            assistant_text = _polish_response_text(assistant_text)
            if _is_calendar_confirmation_required_items(result.items):
                self._pending_calendar_drafts[thread_id] = _extract_calendar_draft_text(
                    items=result.items,
                    fallback=effective_calendar_text,
                )
            elif _is_calendar_created_items(result.items):
                self._pending_calendar_drafts.pop(thread_id, None)
            self._persist_tool_events(
                thread_id=thread_id,
                user_text=text,
                assistant_text=assistant_text,
                tool_name=result.tool_name,
                query=result.query,
                sources=sources,
                authorization=authorization,
                decision_action=route.action,
                capability_label="Google Calendar",
                extra_meta=token_meta,
            )
            return OrchestratorResult(
                response=assistant_text,
                decision=AgentDecision(
                    action=route.action,
                    reason=route.reason,
                    confidence=route.confidence,
                ),
                sources=sources,
            )

        if route.action == "google_gmail":
            tool = self.tool_registry.get("google_gmail")
            effective_gmail_text = text
            if pending_gmail_send and _is_gmail_send_confirmation_reply(text):
                effective_gmail_text = _build_confirmed_gmail_send_request(
                    pending_text=pending_gmail_send,
                    followup_text=text,
                )
            try:
                access_token, token_meta = self._resolve_google_access_token(
                    authorization=authorization,
                    integration_label="Gmail",
                )
            except Exception as exc:
                assistant_text = str(exc)
                self._persist_nonfatal_tool_error_events(
                    thread_id=thread_id,
                    user_text=text,
                    assistant_text=assistant_text,
                    query=effective_gmail_text,
                    authorization=authorization,
                    decision_action="google_gmail",
                    tool_name="google_gmail",
                    capability_label="Gmail",
                )
                return OrchestratorResult(
                    response=assistant_text,
                    decision=AgentDecision(
                        action="google_gmail",
                        reason="google_gmail_auth_failed",
                        confidence=1.0,
                    ),
                    sources=[],
                )

            allowed_domains = [
                value.strip().lower()
                for value in settings.gmail_allowed_recipient_domains.split(",")
                if value.strip()
            ]

            try:
                result = tool.run(
                    ToolContext(
                        thread_id=thread_id,
                        user_text=effective_gmail_text,
                        tool_meta={
                            "access_token": access_token,
                            "max_results": 8,
                            "allowed_recipient_domains": allowed_domains,
                        },
                    )
                )
            except Exception as exc:
                assistant_text = (
                    "I routed this request to Gmail, but the request failed. "
                    f"Error: {exc}"
                )
                self._persist_nonfatal_tool_error_events(
                    thread_id=thread_id,
                    user_text=text,
                    assistant_text=assistant_text,
                    query=effective_gmail_text,
                    authorization=authorization,
                    decision_action="google_gmail",
                    tool_name="google_gmail",
                    capability_label="Gmail",
                )
                return OrchestratorResult(
                    response=assistant_text,
                    decision=AgentDecision(
                        action="google_gmail",
                        reason="google_gmail_failed",
                        confidence=1.0,
                    ),
                    sources=[],
                )

            assistant_text, sources = _format_google_gmail_response(result.items)
            assistant_text = _polish_response_text(assistant_text)
            if _is_gmail_send_confirmation_required_items(result.items):
                self._pending_gmail_send_requests[thread_id] = _extract_gmail_send_pending_text(
                    items=result.items,
                    fallback=effective_gmail_text,
                )
            elif _is_gmail_sent_items(result.items):
                self._pending_gmail_send_requests.pop(thread_id, None)
            self._persist_tool_events(
                thread_id=thread_id,
                user_text=text,
                assistant_text=assistant_text,
                tool_name=result.tool_name,
                query=result.query,
                sources=sources,
                authorization=authorization,
                decision_action=route.action,
                capability_label="Gmail",
                extra_meta=token_meta,
            )
            return OrchestratorResult(
                response=assistant_text,
                decision=AgentDecision(
                    action=route.action,
                    reason=route.reason,
                    confidence=route.confidence,
                ),
                sources=sources,
            )

        if route.action == "google_drive":
            tool = self.tool_registry.get("google_drive")
            try:
                access_token, token_meta = self._resolve_google_access_token(
                    authorization=authorization,
                    integration_label="Google Drive",
                )
            except Exception as exc:
                assistant_text = str(exc)
                self._persist_nonfatal_tool_error_events(
                    thread_id=thread_id,
                    user_text=text,
                    assistant_text=assistant_text,
                    query=text,
                    authorization=authorization,
                    decision_action="google_drive",
                    tool_name="google_drive",
                    capability_label="Google Drive",
                )
                return OrchestratorResult(
                    response=assistant_text,
                    decision=AgentDecision(
                        action="google_drive",
                        reason="google_drive_auth_failed",
                        confidence=1.0,
                    ),
                    sources=[],
                )

            try:
                result = tool.run(
                    ToolContext(
                        thread_id=thread_id,
                        user_text=text,
                        tool_meta={
                            "access_token": access_token,
                            "max_results": 8,
                        },
                    )
                )
            except Exception as exc:
                assistant_text = (
                    "I routed this request to Google Drive, but the request failed. "
                    f"Error: {exc}"
                )
                self._persist_nonfatal_tool_error_events(
                    thread_id=thread_id,
                    user_text=text,
                    assistant_text=assistant_text,
                    query=text,
                    authorization=authorization,
                    decision_action="google_drive",
                    tool_name="google_drive",
                    capability_label="Google Drive",
                )
                return OrchestratorResult(
                    response=assistant_text,
                    decision=AgentDecision(
                        action="google_drive",
                        reason="google_drive_failed",
                        confidence=1.0,
                    ),
                    sources=[],
                )

            assistant_text, sources = _format_google_drive_response(result.items)
            assistant_text = _polish_response_text(assistant_text)
            self._persist_tool_events(
                thread_id=thread_id,
                user_text=text,
                assistant_text=assistant_text,
                tool_name=result.tool_name,
                query=result.query,
                sources=sources,
                authorization=authorization,
                decision_action=route.action,
                capability_label="Google Drive",
                extra_meta=token_meta,
            )
            return OrchestratorResult(
                response=assistant_text,
                decision=AgentDecision(
                    action=route.action,
                    reason=route.reason,
                    confidence=route.confidence,
                ),
                sources=sources,
            )

        tool = self.tool_registry.get("web_search")
        last_web_query = self._last_web_search_query_by_thread.get(thread_id)
        if (_is_generic_web_followup_request(text) or _is_link_followup_request(text)) and last_web_query:
            web_search_text = _build_followup_web_query(
                followup_text=text,
                previous_query=last_web_query,
            )
        elif _is_link_followup_request(text) and previous_non_meta_user_text:
            web_search_text = _build_followup_web_query(
                followup_text=text,
                previous_query=previous_non_meta_user_text,
            )
        elif is_meta_web_request and last_web_query:
            web_search_text = last_web_query
        elif is_meta_web_request and previous_non_meta_user_text:
            web_search_text = previous_non_meta_user_text
        else:
            web_search_text = text
        try:
            result = tool.run(ToolContext(thread_id=thread_id, user_text=web_search_text))
        except Exception as exc:
            assistant_text = (
                "I routed this request to web search, but the search providers failed. "
                f"Error: {exc}"
            )
            self._persist_nonfatal_tool_error_events(
                thread_id=thread_id,
                user_text=text,
                assistant_text=assistant_text,
                query=web_search_text,
                authorization=authorization,
                decision_action="web_search",
                tool_name="web_search",
                capability_label="Web Search",
            )
            return OrchestratorResult(
                response=assistant_text,
                decision=AgentDecision(
                    action="web_search",
                    reason="web_search_failed",
                    confidence=1.0,
                ),
                sources=[],
            )

        assistant_text, sources = _format_web_search_response(result.items, text)
        assistant_text = _polish_response_text(assistant_text)
        assistant_text = _verify_numeric_claims(
            user_text=text,
            assistant_text=assistant_text,
            sources=sources,
        )
        assistant_text = enforce_verification_policy(
            user_text=text,
            assistant_text=assistant_text,
            sources=sources,
            profile=verification,
        )
        self._persist_tool_events(
            thread_id=thread_id,
            user_text=text,
            assistant_text=assistant_text,
            tool_name=result.tool_name,
            query=web_search_text,
            sources=sources,
            authorization=authorization,
            decision_action=route.action,
            capability_label="Web Search",
        )
        self._last_web_search_query_by_thread[thread_id] = web_search_text

        return OrchestratorResult(
            response=assistant_text,
            decision=AgentDecision(
                action=route.action,
                reason=route.reason,
                confidence=route.confidence,
            ),
            sources=sources,
        )

    def _persist_tool_events(
        self,
        thread_id: str,
        user_text: str,
        assistant_text: str,
        tool_name: str,
        query: str,
        sources: list[dict[str, str]],
        authorization: str | None,
        decision_action: str,
        capability_label: str,
        extra_meta: dict[str, object] | None = None,
    ) -> None:
        user_meta = {
            "source": "cortexagent",
            "decision": decision_action,
            "tool_usage": {
                "tool": tool_name,
                "query": query,
            },
        }
        self._add_event_with_retry(
            thread_id=thread_id,
            actor="user",
            content=user_text,
            meta=user_meta,
            authorization=authorization,
        )

        assistant_meta: dict[str, object] = {
            "source": f"cortexagent_{decision_action}",
            "tool": tool_name,
            "query": query,
            "source_urls": [s["url"] for s in sources],
            "tool_usage": {
                "tool": tool_name,
                "query": query,
                "source_count": len(sources),
            },
            "agent_trace": {
                "version": 1,
                "source": "cortex-agent",
                "action": decision_action,
                "capabilities": [
                    {"id": decision_action, "type": "tool", "label": capability_label}
                ],
            },
        }
        if extra_meta:
            assistant_meta["tool_usage"] = {
                **(assistant_meta.get("tool_usage") or {}),
                **extra_meta,
            }

        self._add_event_with_retry(
            thread_id=thread_id,
            actor="assistant",
            content=assistant_text,
            meta=assistant_meta,
            authorization=authorization,
        )

    def _add_event_with_retry(
        self,
        thread_id: str,
        actor: str,
        content: str,
        meta: dict[str, object],
        authorization: str | None,
    ) -> None:
        safe_content = _clamp_event_content(content)
        attempts = 0
        last_error: Exception | None = None
        while attempts < 2:
            attempts += 1
            try:
                self.ltm_client.add_event(
                    thread_id=thread_id,
                    actor=actor,
                    content=safe_content,
                    meta=meta,
                    authorization=authorization,
                )
                return
            except Exception as exc:
                last_error = exc
        raise RuntimeError(
            f"Failed to persist {actor} event for tool response after retry: {last_error}"
        )

    def _persist_nonfatal_tool_error_events(
        self,
        thread_id: str,
        user_text: str,
        assistant_text: str,
        query: str,
        authorization: str | None,
        decision_action: str,
        tool_name: str,
        capability_label: str,
    ) -> None:
        try:
            self._persist_tool_events(
                thread_id=thread_id,
                user_text=user_text,
                assistant_text=assistant_text,
                tool_name=tool_name,
                query=query,
                sources=[],
                authorization=authorization,
                decision_action=decision_action,
                capability_label=capability_label,
            )
        except Exception:
            # Keep tool failure response path resilient even if event persistence is unavailable.
            return

    def _resolve_google_access_token(
        self,
        authorization: str | None,
        integration_label: str,
    ) -> tuple[str, dict[str, object]]:
        if not self.connected_accounts_repo or not self.google_oauth:
            raise RuntimeError(f"{integration_label} is currently unavailable.")
        if not authorization:
            raise RuntimeError(f"Please sign in, then reconnect {integration_label}.")

        user_id = resolve_user_id_from_authorization(
            authorization=authorization,
            supabase_url=settings.supabase_url,
            supabase_anon_key=settings.supabase_anon_key,
            timeout_seconds=5,
        )
        resolved = self.connected_accounts_repo.resolve_provider_token(
            user_id=user_id, provider="google"
        )
        if resolved is None:
            raise RuntimeError(
                f"{integration_label} is not connected. Use Connect Google in Settings first."
            )
        if resolved.access_token and not resolved.is_access_token_expired:
            return (
                resolved.access_token,
                {
                    "token_refreshed": False,
                    "token_expires_at": resolved.expires_at.isoformat()
                    if resolved.expires_at
                    else None,
                },
            )

        if not resolved.refresh_token:
            raise RuntimeError(
                "Google connection expired and cannot refresh automatically. "
                f"Please reconnect {integration_label}."
            )
        try:
            refreshed = self.google_oauth.refresh_access_token(resolved.refresh_token)
        except Exception as exc:
            raise RuntimeError(
                "Google authorization expired and refresh failed. "
                f"Please reconnect {integration_label}."
            ) from exc

        next_refresh = refreshed.refresh_token or resolved.refresh_token
        expires_at = None
        if isinstance(refreshed.expires_in, int) and refreshed.expires_in > 0:
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=refreshed.expires_in)

        self.connected_accounts_repo.upsert_active_account(
            payload=ConnectedAccountUpsert(
                user_id=resolved.account.user_id,
                provider=resolved.account.provider,
                provider_account_id=resolved.account.provider_account_id,
                access_token=refreshed.access_token,
                refresh_token=next_refresh,
                token_type=refreshed.token_type or resolved.account.token_type,
                scope=refreshed.scope or resolved.account.scope,
                expires_at=expires_at,
                status="active",
                meta=resolved.account.meta,
            )
        )
        return (
            refreshed.access_token,
            {
                "token_refreshed": True,
                "token_expires_at": expires_at.isoformat() if expires_at else None,
            },
        )

    def _run_multi_step_pipeline(
        self,
        thread_id: str,
        text: str,
        authorization: str | None,
        steps: list[OrchestrationStep],
    ) -> OrchestratorResult:
        step_results: list[OrchestrationStepResult] = []
        for step in steps:
            result = self._execute_pipeline_step(
                thread_id=thread_id,
                user_text=text,
                step=step,
                authorization=authorization,
            )
            step_results.append(result)
            if result.success and result.action in {"google_calendar", "google_gmail", "google_drive"}:
                self._last_tool_action_by_thread[thread_id] = result.action
            if result.success and result.action == "web_search":
                self._last_web_search_query_by_thread[thread_id] = result.query

        combined_sources = _dedupe_sources_by_url(
            [source for step in step_results for source in step.sources]
        )
        response = _format_orchestration_response(step_results=step_results)
        decision = AgentDecision(
            action="orchestration",
            reason="multi_step_pipeline_executed",
            confidence=0.93,
        )
        self._persist_orchestration_events(
            thread_id=thread_id,
            user_text=text,
            assistant_text=response,
            authorization=authorization,
            step_results=step_results,
            sources=combined_sources,
        )
        self._last_pipeline_results_by_thread[thread_id] = [
            _serialize_orchestration_step_result(step) for step in step_results
        ]
        return OrchestratorResult(
            response=response,
            decision=decision,
            sources=combined_sources,
            tool_pipeline=[
                _serialize_orchestration_step_result(step) for step in step_results
            ],
        )

    def _execute_pipeline_step(
        self,
        thread_id: str,
        user_text: str,
        step: OrchestrationStep,
        authorization: str | None,
    ) -> OrchestrationStepResult:
        action = step.action
        query = step.query
        if action == "google_calendar":
            tool_name = "google_calendar"
            capability_label = "Google Calendar"
            try:
                access_token, token_meta = self._resolve_google_access_token(
                    authorization=authorization,
                    integration_label=capability_label,
                )
            except Exception as exc:
                return OrchestrationStepResult(
                    action=action,
                    tool_name=tool_name,
                    query=query,
                    assistant_text=str(exc),
                    sources=[],
                    success=False,
                    reason="google_calendar_auth_failed",
                    capability_label=capability_label,
                )
            try:
                tool = self.tool_registry.get(tool_name)
                result = tool.run(
                    ToolContext(
                        thread_id=thread_id,
                        user_text=query,
                        tool_meta={"access_token": access_token, "max_results": 8},
                    )
                )
                assistant_text, sources = _format_google_calendar_response(result.items)
                assistant_text = _polish_response_text(assistant_text)
                if _is_calendar_confirmation_required_items(result.items):
                    self._pending_calendar_drafts[thread_id] = _extract_calendar_draft_text(
                        items=result.items,
                        fallback=query,
                    )
                elif _is_calendar_created_items(result.items):
                    self._pending_calendar_drafts.pop(thread_id, None)
                return OrchestrationStepResult(
                    action=action,
                    tool_name=result.tool_name,
                    query=result.query,
                    assistant_text=assistant_text,
                    sources=sources,
                    success=True,
                    reason="ok",
                    capability_label=capability_label,
                    extra_meta=token_meta,
                )
            except Exception as exc:
                return OrchestrationStepResult(
                    action=action,
                    tool_name=tool_name,
                    query=query,
                    assistant_text=(
                        "I routed this step to Google Calendar, but it failed. "
                        f"Error: {exc}"
                    ),
                    sources=[],
                    success=False,
                    reason="google_calendar_failed",
                    capability_label=capability_label,
                )

        if action == "google_gmail":
            tool_name = "google_gmail"
            capability_label = "Gmail"
            try:
                access_token, token_meta = self._resolve_google_access_token(
                    authorization=authorization,
                    integration_label=capability_label,
                )
            except Exception as exc:
                return OrchestrationStepResult(
                    action=action,
                    tool_name=tool_name,
                    query=query,
                    assistant_text=str(exc),
                    sources=[],
                    success=False,
                    reason="google_gmail_auth_failed",
                    capability_label=capability_label,
                )
            allowed_domains = [
                value.strip().lower()
                for value in settings.gmail_allowed_recipient_domains.split(",")
                if value.strip()
            ]
            try:
                tool = self.tool_registry.get(tool_name)
                result = tool.run(
                    ToolContext(
                        thread_id=thread_id,
                        user_text=query,
                        tool_meta={
                            "access_token": access_token,
                            "max_results": 8,
                            "allowed_recipient_domains": allowed_domains,
                        },
                    )
                )
                assistant_text, sources = _format_google_gmail_response(result.items)
                assistant_text = _polish_response_text(assistant_text)
                if _is_gmail_send_confirmation_required_items(result.items):
                    self._pending_gmail_send_requests[thread_id] = _extract_gmail_send_pending_text(
                        items=result.items,
                        fallback=query,
                    )
                elif _is_gmail_sent_items(result.items):
                    self._pending_gmail_send_requests.pop(thread_id, None)
                return OrchestrationStepResult(
                    action=action,
                    tool_name=result.tool_name,
                    query=result.query,
                    assistant_text=assistant_text,
                    sources=sources,
                    success=True,
                    reason="ok",
                    capability_label=capability_label,
                    extra_meta=token_meta,
                )
            except Exception as exc:
                return OrchestrationStepResult(
                    action=action,
                    tool_name=tool_name,
                    query=query,
                    assistant_text=(
                        "I routed this step to Gmail, but it failed. "
                        f"Error: {exc}"
                    ),
                    sources=[],
                    success=False,
                    reason="google_gmail_failed",
                    capability_label=capability_label,
                )

        if action == "google_drive":
            tool_name = "google_drive"
            capability_label = "Google Drive"
            try:
                access_token, token_meta = self._resolve_google_access_token(
                    authorization=authorization,
                    integration_label=capability_label,
                )
            except Exception as exc:
                return OrchestrationStepResult(
                    action=action,
                    tool_name=tool_name,
                    query=query,
                    assistant_text=str(exc),
                    sources=[],
                    success=False,
                    reason="google_drive_auth_failed",
                    capability_label=capability_label,
                )
            try:
                tool = self.tool_registry.get(tool_name)
                result = tool.run(
                    ToolContext(
                        thread_id=thread_id,
                        user_text=query,
                        tool_meta={"access_token": access_token, "max_results": 8},
                    )
                )
                assistant_text, sources = _format_google_drive_response(result.items)
                assistant_text = _polish_response_text(assistant_text)
                return OrchestrationStepResult(
                    action=action,
                    tool_name=result.tool_name,
                    query=result.query,
                    assistant_text=assistant_text,
                    sources=sources,
                    success=True,
                    reason="ok",
                    capability_label=capability_label,
                    extra_meta=token_meta,
                )
            except Exception as exc:
                return OrchestrationStepResult(
                    action=action,
                    tool_name=tool_name,
                    query=query,
                    assistant_text=(
                        "I routed this step to Google Drive, but it failed. "
                        f"Error: {exc}"
                    ),
                    sources=[],
                    success=False,
                    reason="google_drive_failed",
                    capability_label=capability_label,
                )

        if action == "web_search":
            tool_name = "web_search"
            capability_label = "Web Search"
            try:
                tool = self.tool_registry.get(tool_name)
                result = tool.run(ToolContext(thread_id=thread_id, user_text=query))
                assistant_text, sources = _format_web_search_response(result.items, user_text)
                assistant_text = _polish_response_text(assistant_text)
                assistant_text = _verify_numeric_claims(
                    user_text=user_text,
                    assistant_text=assistant_text,
                    sources=sources,
                )
                return OrchestrationStepResult(
                    action=action,
                    tool_name=result.tool_name,
                    query=result.query,
                    assistant_text=assistant_text,
                    sources=sources,
                    success=True,
                    reason="ok",
                    capability_label=capability_label,
                    extra_meta=None,
                )
            except Exception as exc:
                return OrchestrationStepResult(
                    action=action,
                    tool_name=tool_name,
                    query=query,
                    assistant_text=(
                        "I routed this step to web search, but it failed. "
                        f"Error: {exc}"
                    ),
                    sources=[],
                    success=False,
                    reason="web_search_failed",
                    capability_label=capability_label,
                )

        return OrchestrationStepResult(
            action=action,
            tool_name=action,
            query=query,
            assistant_text=f"Unsupported orchestration step action: {action}",
            sources=[],
            success=False,
            reason="unsupported_step_action",
            capability_label=_capability_label_for_action(action),
        )

    def _persist_orchestration_events(
        self,
        thread_id: str,
        user_text: str,
        assistant_text: str,
        authorization: str | None,
        step_results: list[OrchestrationStepResult],
        sources: list[dict[str, str]],
    ) -> None:
        try:
            user_meta = {
                "source": "cortexagent",
                "decision": "orchestration",
                "tool_pipeline": {
                    "step_count": len(step_results),
                    "steps": [
                        {
                            "action": step.action,
                            "tool": step.tool_name,
                            "query": step.query,
                            "success": step.success,
                            "reason": step.reason,
                        }
                        for step in step_results
                    ],
                },
            }
            self._add_event_with_retry(
                thread_id=thread_id,
                actor="user",
                content=user_text,
                meta=user_meta,
                authorization=authorization,
            )
            assistant_meta = {
                "source": "cortexagent_orchestration",
                "tool_pipeline": user_meta["tool_pipeline"],
                "source_urls": [source.get("url", "") for source in sources if source.get("url")],
                "agent_trace": {
                    "version": 1,
                    "source": "cortex-agent",
                    "action": "orchestration",
                    "capabilities": [
                        {
                            "id": step.action,
                            "type": "tool",
                            "label": step.capability_label,
                        }
                        for step in step_results
                    ],
                },
            }
            self._add_event_with_retry(
                thread_id=thread_id,
                actor="assistant",
                content=assistant_text,
                meta=assistant_meta,
                authorization=authorization,
            )
        except Exception:
            return


def _format_web_search_response(
    items: list, user_text: str
) -> tuple[str, list[dict[str, str]]]:
    if not items:
        return (
            "I could not find reliable web results for that query right now.",
            [],
        )

    timestamp = _friendly_local_timestamp()
    raw_sources: list[dict[str, str]] = []
    for item in items[:8]:
        snippet = item.snippet.strip()
        raw_sources.append({"title": item.title, "url": item.url, "snippet": snippet})

    sources = _select_clean_sources(raw_sources, user_text=user_text, max_count=3)

    list_limit = 3
    if _looks_like_live_price_query(user_text):
        subject = _infer_price_subject(user_text)
        spot_price, spot_source = _fetch_crypto_spot_price_usd(user_text)
        values = _extract_money_values_from_sources(sources)
        if spot_price is not None:
            headline = f"As of {timestamp}, {subject} is about ${spot_price:,.2f} USD."
        elif values:
            low = min(values)
            high = max(values)
            mid = values[len(values) // 2]
            if (high - low) / max(mid, 1.0) <= 0.006:
                headline = f"As of {timestamp}, {subject} is about ${mid:,.2f} USD."
            else:
                headline = (
                    f"As of {timestamp}, {subject} appears between "
                    f"${low:,.2f} and ${high:,.2f} USD across sources."
                )
        else:
            headline = (
                f"As of {timestamp}, I could not extract a stable live {subject} price "
                "from the available snippets."
            )
        lines = [
            "Live Price Snapshot",
            "-------------------",
            headline,
            "",
            "Source Links",
            "------------",
        ]
        if spot_source:
            lines.append(f"- CoinGecko spot quote | {spot_source}")
            list_limit = 2
    else:
        lines = [
            "Web Results",
            "-----------",
            f"Checked {len(sources)} sources at {timestamp}.",
            "",
            "Top Matches",
            "-----------",
        ]
        for idx, src in enumerate(sources, start=1):
            snippet = src.get("snippet", "")
            title = src.get("title", "")
            lines.append(f"{idx}. {title}")
            if snippet:
                lines.append(f"   {snippet}")
            lines.append("")
        lines.extend(["Source Links", "------------"])

    for src in sources[:list_limit]:
        lines.append(f"- {src['title']} | {src['url']}")

    return "\n".join(lines), sources


def _clamp_event_content(content: str, max_len: int = 5800) -> str:
    text = (content or "").strip()
    if len(text) <= max_len:
        return text
    clipped = text[:max_len].rstrip()
    return f"{clipped}\n\n[Truncated for event storage limit]"


def _format_google_calendar_response(
    items: list,
) -> tuple[str, list[dict[str, str]]]:
    if not items:
        return (
            "I could not find upcoming events in your Google Calendar.",
            [],
        )
    if (
        len(items) == 1
        and isinstance(getattr(items[0], "title", None), str)
        and str(getattr(items[0], "title", "")).strip().lower() == "confirmation required"
    ):
        snippet = (getattr(items[0], "snippet", "") or "").strip()
        message = snippet or (
            "Confirmation required before writing to Google Calendar. "
            "Reply with 'confirm' to proceed or 'cancel' to stop."
        )
        return (
            message,
            [
                {
                    "title": "Google Calendar",
                    "url": "https://calendar.google.com/",
                    "snippet": "Write confirmation required",
                }
            ],
        )

    has_created = any(
        isinstance(getattr(item, "title", None), str)
        and str(getattr(item, "title", "")).startswith("[Created] ")
        for item in items
    )
    lines = (
        ["Google Calendar updated. Upcoming events:"]
        if has_created
        else ["Upcoming Google Calendar events:"]
    )
    sources: list[dict[str, str]] = []
    for idx, item in enumerate(items[:8], start=1):
        title = item.title
        item_url = (item.url or "").strip() or "https://calendar.google.com/"
        if title.startswith("[Created] "):
            title = title.replace("[Created] ", "", 1) + " (created)"
        pretty_snippet = _normalize_calendar_snippet(item.snippet)
        starts_text, location_text = _split_calendar_snippet_fields(pretty_snippet)
        lines.append(f"{idx}. {title}")
        lines.append(f"   {starts_text}")
        if location_text:
            lines.append(f"   Location: {location_text}")
        lines.append(f"   Link: {item_url}")
        sources.append(
            {
                "title": title,
                "url": item_url,
                "snippet": pretty_snippet,
            }
        )
    return ("\n".join(lines), sources)


def _format_google_gmail_response(
    items: list,
) -> tuple[str, list[dict[str, str]]]:
    if not items:
        return ("I could not find Gmail results for that request.", [])

    if (
        len(items) == 1
        and str(getattr(items[0], "title", "")).strip().lower()
        == "send confirmation required"
    ):
        snippet = str(getattr(items[0], "snippet", "")).strip()
        return (
            snippet
            or "Confirmation required before sending Gmail draft. Reply with confirm or cancel.",
            [
                {
                    "title": "Gmail",
                    "url": "https://mail.google.com/",
                    "snippet": "Send confirmation required",
                }
            ],
        )

    has_send = any(
        str(getattr(item, "title", "")).strip().lower().startswith("[sent]")
        for item in items
    )
    has_draft = any(
        str(getattr(item, "title", "")).strip().lower().startswith("[drafted]")
        for item in items
    )
    if has_send:
        lines = ["Gmail sent successfully:"]
    elif has_draft:
        lines = ["Gmail draft created:"]
    else:
        lines = ["Primary inbox emails:"]

    sources: list[dict[str, str]] = []
    for idx, item in enumerate(items[:8], start=1):
        raw_title = str(getattr(item, "title", "")).strip()
        pretty_title = _normalize_gmail_result_title(raw_title)
        item_url = str(getattr(item, "url", "")).strip() or "https://mail.google.com/"
        snippet = str(getattr(item, "snippet", "")).strip()
        lines.append(f"{idx}. {pretty_title}")
        for detail in _format_gmail_item_details(snippet):
            lines.append(f"   {detail}")
        lines.append(f"   Link: {item_url}")
        lines.append("")
        sources.append(
            {
                "title": raw_title or "Gmail result",
                "url": item_url,
                "snippet": snippet,
            }
        )
    if lines and not lines[-1].strip():
        lines.pop()
    return ("\n".join(lines), sources)


def _format_google_drive_response(
    items: list,
) -> tuple[str, list[dict[str, str]]]:
    if not items:
        return ("I could not find Google Drive results for that request.", [])

    lines = ["Google Drive results:"]
    sources: list[dict[str, str]] = []
    for idx, item in enumerate(items, start=1):
        title = str(getattr(item, "title", "")).strip() or f"File {idx}"
        snippet = str(getattr(item, "snippet", "")).strip()
        item_url = str(getattr(item, "url", "")).strip() or "https://drive.google.com/drive/my-drive"
        lines.append(f"{idx}. {title}")
        if snippet:
            lines.append(f"   {snippet}")
        lines.append(f"   Link: {item_url}")
        sources.append(
            {
                "title": title,
                "url": item_url,
                "snippet": snippet,
            }
        )
    return ("\n".join(lines), sources)


def _split_calendar_snippet_fields(snippet: str) -> tuple[str, str]:
    parts = [part.strip() for part in (snippet or "").split("|", 1)]
    starts_text = parts[0] if parts and parts[0] else "Starts: Time unavailable"
    location_text = parts[1] if len(parts) > 1 else ""
    return starts_text, location_text


def _normalize_gmail_result_title(raw_title: str) -> str:
    title = (raw_title or "").strip()
    if not title:
        return "Gmail result"
    thread_match = re.match(r"^Thread\s+([a-z0-9_-]+)\s*\|\s*(.+)$", title, re.IGNORECASE)
    if thread_match:
        subject = thread_match.group(2).strip()
        return subject
    read_match = re.match(
        r"^Message from thread\s+([a-z0-9_-]+)\s*\|\s*(.+)$",
        title,
        re.IGNORECASE,
    )
    if read_match:
        subject = read_match.group(2).strip()
        return subject
    drafted_match = re.match(r"^\[Drafted\]\s+Thread\s+([a-z0-9_-]+)$", title, re.IGNORECASE)
    if drafted_match:
        return "Drafted reply"
    return title


def _format_gmail_item_details(snippet: str) -> list[str]:
    expanded = _expand_tool_item_details(snippet)
    if not expanded:
        return []
    first = expanded[0]
    if first.lower().startswith("from:"):
        sender = first[5:].strip() or "Unknown sender"
        preview_rows: list[str] = []
        security_rows: list[str] = []
        for row in expanded[1:]:
            lowered = row.lower()
            if lowered.startswith("[security]"):
                security_rows.append(row)
                continue
            preview_rows.append(row)
        out = [f"From: {sender}"]
        if preview_rows:
            readable_rows = _filter_gmail_body_noise(preview_rows)
            if not readable_rows:
                readable_rows = [
                    "This message body is mostly links/buttons. Open in Gmail for full formatted content."
                ]
            if len(readable_rows) == 1:
                out.append(f"Preview: {readable_rows[0]}")
            else:
                out.append(f"Body: {readable_rows[0]}")
                out.extend(readable_rows[1:])
        out.extend(security_rows)
        return out
    return expanded


def _filter_gmail_body_noise(rows: list[str]) -> list[str]:
    out: list[str] = []
    for row in rows:
        line = row.strip()
        if not line:
            continue
        if _is_url_noise_line(line):
            continue
        out.append(line)
    return out[:12]


def _is_url_noise_line(line: str) -> bool:
    lowered = line.lower()
    if lowered.startswith("http://") or lowered.startswith("https://"):
        return True
    if "linkedin.com/comm/jobs" in lowered:
        return True
    if "trackingid=" in lowered or "trk" in lowered or "utm_" in lowered:
        return True
    url_count = len(re.findall(r"https?://", line, flags=re.IGNORECASE))
    if url_count >= 1 and len(line) > 90:
        return True
    # "View job: https://..." style call-to-action rows are noisy in plain text cards.
    if re.match(r"^[a-z][a-z\s]{2,40}:\s*https?://", lowered):
        return True
    return False


def _expand_tool_item_details(snippet: str) -> list[str]:
    if not snippet:
        return []
    out: list[str] = []
    for row in snippet.splitlines():
        line = row.strip()
        if not line:
            continue
        if "|" in line:
            for part in line.split("|"):
                cleaned = part.strip()
                if cleaned:
                    out.append(cleaned)
            continue
        out.append(line)
    return out


def _normalize_calendar_snippet(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return "Time unavailable"
    parts = [part.strip() for part in text.split("|") if part.strip()]
    time_text = ""
    location_parts: list[str] = []
    for part in parts:
        lowered = part.lower()
        if lowered.startswith("starts:") and not time_text:
            time_text = part
            continue
        if lowered.startswith("created event"):
            continue
        location_parts.append(part)
    if not time_text:
        time_text = parts[0] if parts else text
        if parts:
            location_parts = parts[1:]
    location = " | ".join(location_parts).strip()
    pretty_time = _humanize_calendar_time_label(time_text)
    if location:
        return f"{pretty_time} | {location}"
    return pretty_time


def _humanize_calendar_time_label(label: str) -> str:
    raw = label.strip()
    if not raw.lower().startswith("starts:"):
        return raw
    value = raw[7:].strip()
    if not value:
        return "Starts: Time unavailable"
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return f"Starts: {value}"
    if parsed.tzinfo is not None:
        stamp = parsed.strftime("%a, %b %d at %I:%M %p").replace(" 0", " ")
        return f"Starts: {stamp}"
    stamp = parsed.strftime("%a, %b %d (all day)").replace(" 0", " ")
    return f"Starts: {stamp}"


PRICE_QUERY_TOKENS = {
    "live price",
    "current price",
    "quote",
    "market price",
    "trading at",
    "usd",
    "eur",
}


def _verify_numeric_claims(
    user_text: str,
    assistant_text: str,
    sources: list[dict[str, str]],
) -> str:
    if not assistant_text.strip() or not sources:
        return assistant_text
    # Price responses are already rendered in a source-grounded template.
    # Avoid prepending a second correction block.
    if _looks_like_live_price_query(user_text):
        return assistant_text

    source_values = _extract_money_values_from_sources(sources)
    response_values = _extract_money_values(assistant_text)
    if not source_values or not response_values:
        return assistant_text
    if not _has_numeric_mismatch(response_values, source_values):
        return assistant_text

    low = min(source_values)
    high = max(source_values)
    mid = source_values[len(source_values) // 2]
    timestamp = _friendly_local_timestamp()

    if (high - low) / max(mid, 1.0) <= 0.006:
        corrected = f"As of {timestamp}, cited sources cluster around ${mid:,.2f}."
    else:
        corrected = (
            f"As of {timestamp}, cited sources vary between ${low:,.2f} and ${high:,.2f}."
        )

    return (
        f"{corrected} Source snippets disagree, so I am showing a range instead of a single "
        "exact live price.\n\n"
        f"{assistant_text}"
    )


def _looks_like_live_price_query(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    if not normalized:
        return False
    has_price_phrase = any(token in normalized for token in PRICE_QUERY_TOKENS)
    has_generic_price_word = bool(re.search(r"\bprice\b", normalized))
    if not (has_price_phrase or has_generic_price_word):
        return False
    if _looks_like_shopping_price_query(normalized):
        return False
    return _looks_like_finance_price_subject(normalized)


def _looks_like_finance_price_subject(normalized_text: str) -> bool:
    return bool(
        re.search(
            r"\b("
            r"bitcoin|btc|ethereum|eth|xrp|solana|sol|dogecoin|doge|cardano|ada|"
            r"crypto|cryptocurrency|stock|stocks|share price|forex|fx|gold|silver|oil|"
            r"nasdaq|dow|s&p|sp500|coin"
            r")\b",
            normalized_text,
        )
    )


def _looks_like_shopping_price_query(normalized_text: str) -> bool:
    return bool(
        re.search(
            r"\b("
            r"camera|cameras|hotel|hotels|flight|flights|laptop|phone|iphone|android|"
            r"headphones|tv|monitor|bike|car|gift|watch|watches|shoe|shoes|bag|mic|microphone"
            r")\b",
            normalized_text,
        )
    )


def _polish_response_text(text: str) -> str:
    if not text:
        return text
    out = text
    out = out.replace("\r\n", "\n")
    out = re.sub(r"\*\*(.*?)\*\*", r"\1", out)
    out = re.sub(r"__([^_]+)__", r"\1", out)
    out = re.sub(r"(^|\n)(\d+)\.([A-Za-z])", r"\1\2. \3", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _infer_price_subject(text: str) -> str:
    lowered = text.strip().lower()
    known_assets = [
        "bitcoin",
        "btc",
        "ethereum",
        "eth",
        "xrp",
        "solana",
        "sol",
        "dogecoin",
        "doge",
        "cardano",
        "ada",
    ]
    for asset in known_assets:
        if re.search(rf"\b{re.escape(asset)}\b", lowered):
            return asset.upper() if len(asset) <= 4 else asset
    return "the asset"


_COINGECKO_ID_BY_ALIAS = {
    "btc": "bitcoin",
    "bitcoin": "bitcoin",
    "eth": "ethereum",
    "ethereum": "ethereum",
    "xrp": "ripple",
    "sol": "solana",
    "solana": "solana",
    "doge": "dogecoin",
    "dogecoin": "dogecoin",
    "ada": "cardano",
    "cardano": "cardano",
}

_COINBASE_PRODUCT_BY_ALIAS = {
    "btc": "BTC-USD",
    "bitcoin": "BTC-USD",
    "eth": "ETH-USD",
    "ethereum": "ETH-USD",
    "xrp": "XRP-USD",
    "sol": "SOL-USD",
    "solana": "SOL-USD",
    "doge": "DOGE-USD",
    "dogecoin": "DOGE-USD",
    "ada": "ADA-USD",
    "cardano": "ADA-USD",
}


def _fetch_crypto_spot_price_usd(user_text: str) -> tuple[float | None, str | None]:
    lowered = user_text.lower()
    alias = None
    for key in _COINGECKO_ID_BY_ALIAS.keys():
        if re.search(rf"\b{re.escape(key)}\b", lowered):
            alias = key
            break
    if not alias:
        return (None, None)

    coinbase_price = _fetch_coinbase_spot_price_usd(alias)
    if coinbase_price is not None:
        product = _COINBASE_PRODUCT_BY_ALIAS[alias]
        return (coinbase_price, f"https://www.coinbase.com/price/{product.split('-')[0].lower()}")

    coin_id = _COINGECKO_ID_BY_ALIAS[alias]
    url = (
        "https://api.coingecko.com/api/v3/simple/price"
        f"?ids={coin_id}&vs_currencies=usd"
    )
    req = urlrequest.Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urlrequest.urlopen(req, timeout=5) as res:
            payload = json.loads(res.read().decode("utf-8"))
    except Exception:
        return (None, None)

    row = payload.get(coin_id, {}) if isinstance(payload, dict) else {}
    value = row.get("usd") if isinstance(row, dict) else None
    if not isinstance(value, (int, float)) or value <= 0:
        return (None, None)
    source_url = f"https://www.coingecko.com/en/coins/{coin_id}"
    return (float(value), source_url)


def _fetch_coinbase_spot_price_usd(alias: str) -> float | None:
    product = _COINBASE_PRODUCT_BY_ALIAS.get(alias)
    if not product:
        return None
    url = f"https://api.coinbase.com/v2/prices/{product}/spot"
    req = urlrequest.Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urlrequest.urlopen(req, timeout=4) as res:
            payload = json.loads(res.read().decode("utf-8"))
    except Exception:
        return None
    data = payload.get("data", {}) if isinstance(payload, dict) else {}
    raw_amount = data.get("amount") if isinstance(data, dict) else None
    if not isinstance(raw_amount, str):
        return None
    try:
        value = float(raw_amount)
    except ValueError:
        return None
    return value if value > 0 else None


def _extract_money_values_from_sources(sources: list[dict[str, str]]) -> list[float]:
    values: list[float] = []
    for src in sources:
        snippet = src.get("snippet", "")
        title = src.get("title", "")
        values.extend(_extract_money_values(snippet))
        values.extend(_extract_money_values(title))
    values.sort()
    return values[:24]


def _extract_money_values(text: str) -> list[float]:
    # Extract likely *price* values; avoid volume/market-cap numbers.
    pattern = re.compile(
        r"(?i)(?:\$\s*|usd\s*)"
        r"([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)"
        r"(?:\s*usd)?"
    )
    blocked_context = {
        "volume",
        "24-hour",
        "24 hour",
        "market cap",
        "fully diluted",
        "circulating",
        "trading volume",
    }
    out: list[float] = []
    for match in pattern.finditer(text):
        raw = match.group(1).replace(",", "")
        start, end = match.span()
        window = text[max(0, start - 28) : min(len(text), end + 28)].lower()
        if any(token in window for token in blocked_context):
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if value <= 0:
            continue
        if value > 1_000_000:
            continue
        out.append(value)
    return out


def _has_numeric_mismatch(response_values: list[float], source_values: list[float]) -> bool:
    tolerance_pct = 0.02
    tolerance_abs = 2.0
    for response_value in response_values:
        nearest = min(source_values, key=lambda source_value: abs(source_value - response_value))
        diff = abs(nearest - response_value)
        if diff <= tolerance_abs:
            continue
        if diff / max(nearest, 1.0) <= tolerance_pct:
            continue
        return True
    return False


_TRACKING_KEYS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "msclkid",
    "gclid",
    "fbclid",
    "igshid",
    "ref",
    "ref_src",
    "source",
    "ad_domain",
    "ad_provider",
    "ad_type",
    "click_metadata",
    "rut",
    "u",
    "u3",
    "rlid",
    "vqd",
    "iurl",
    "cid",
    "id",
    "ig",
}

_PRICE_SOURCE_HOST_PREFERENCE = [
    "coinmarketcap.com",
    "coindesk.com",
    "google.com",
    "finance.yahoo.com",
    "coinbase.com",
]


def _select_clean_sources(
    sources: list[dict[str, str]], user_text: str, max_count: int
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen_hosts: set[str] = set()
    price_mode = _looks_like_live_price_query(user_text)

    ordered = sources
    if price_mode:
        ordered = sorted(sources, key=lambda s: _source_rank_for_price(s.get("url", "")))

    for src in ordered:
        raw_url = src.get("url", "")
        cleaned_url = _clean_source_url(raw_url)
        if not cleaned_url:
            continue
        host = (urlparse(cleaned_url).netloc or "").lower()
        if not host or host in seen_hosts:
            continue
        title = (src.get("title", "") or "").strip()
        if _looks_like_ad_source(title, cleaned_url):
            continue
        seen_hosts.add(host)
        out.append(
            {
                "title": title or host,
                "url": cleaned_url,
                "snippet": (src.get("snippet", "") or "").strip(),
            }
        )
        if len(out) >= max_count:
            break

    return out


def _source_rank_for_price(url: str) -> tuple[int, str]:
    host = (urlparse(url).netloc or "").lower()
    for idx, preferred in enumerate(_PRICE_SOURCE_HOST_PREFERENCE):
        if preferred in host:
            return (idx, host)
    return (len(_PRICE_SOURCE_HOST_PREFERENCE) + 1, host)


def _looks_like_ad_source(title: str, url: str) -> bool:
    lowered_title = title.lower()
    lowered_url = url.lower()
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "/").lower()
    ad_markers = [
        "trusted",
        "sign up",
        "buy now",
        "easy cryptocurrency trading",
        "advert",
    ]
    generic_homepage_markers = [
        "google",
        "yandex",
        "duckduckgo",
        "bing",
        "search",
    ]
    if any(marker in lowered_title for marker in ad_markers):
        return True
    if path in {"", "/"} and any(marker in lowered_title for marker in generic_homepage_markers):
        return True
    if host in {"google.com", "www.google.com", "yandex.com", "www.yandex.com"} and path in {"", "/"}:
        return True
    if "duckduckgo.com/y.js" in lowered_url or "bing.com/aclick" in lowered_url:
        return True
    return False


def _clean_source_url(raw_url: str) -> str:
    parsed = urlparse((raw_url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""

    host = parsed.netloc.lower()
    path = parsed.path or "/"
    if ("duckduckgo.com" in host and path == "/y.js") or ("bing.com" in host and path == "/aclick"):
        return ""

    kept = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=False):
        if key.lower() in _TRACKING_KEYS:
            continue
        kept.append((key, value))

    query = urlencode(kept, doseq=True)
    cleaned = parsed._replace(query=query, fragment="")
    normalized = urlunparse(cleaned)
    return normalized


def _friendly_local_timestamp() -> str:
    local_now = datetime.now().astimezone()
    stamp = local_now.strftime("%b %d, %Y at %I:%M %p %Z")
    return stamp.replace(" at 0", " at ")


def _build_multi_step_plan(text: str) -> list[OrchestrationStep]:
    lowered = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not lowered:
        return []

    gmail_requested = bool(
        re.search(r"\b(email|gmail|inbox|thread|reply|draft|send)\b", lowered)
    )
    drive_requested = bool(
        re.search(r"\b(drive|doc|document|file|folder|attach)\b", lowered)
    )
    calendar_requested = bool(
        re.search(r"\b(calendar|meeting|schedule|event|reschedule|remind|availability)\b", lowered)
    )
    web_requested = bool(
        re.search(r"\b(web|search|look up|lookup|latest|news|online)\b", lowered)
    )

    requested_actions: list[str] = []
    if gmail_requested:
        requested_actions.append("google_gmail")
    if drive_requested:
        requested_actions.append("google_drive")
    if calendar_requested:
        requested_actions.append("google_calendar")
    if web_requested:
        requested_actions.append("web_search")

    if len(requested_actions) < 2:
        return []

    ordered = ["google_gmail", "google_drive", "google_calendar", "web_search"]
    if gmail_requested and calendar_requested:
        calendar_pos = _first_keyword_position(
            lowered,
            [r"\bcalendar\b", r"\bmeeting\b", r"\bschedule\b", r"\bevent\b"],
        )
        gmail_pos = _first_keyword_position(
            lowered,
            [r"\bgmail\b", r"\bemail\b", r"\binbox\b"],
        )
        if (
            calendar_pos >= 0
            and gmail_pos >= 0
            and calendar_pos < gmail_pos
            and re.search(r"\b(then|after|followed by|next)\b", lowered)
        ):
            ordered = ["google_calendar", "google_gmail", "google_drive", "web_search"]
    steps: list[OrchestrationStep] = []
    for action in ordered:
        if action not in requested_actions:
            continue
        steps.append(
            OrchestrationStep(
                action=action,
                query=_build_step_query(action=action, user_text=text),
            )
        )
    return steps


def _build_step_query(action: str, user_text: str) -> str:
    lowered = re.sub(r"\s+", " ", (user_text or "").strip().lower())
    if action == "google_gmail":
        email_match = re.search(
            r"\bemail\s+(?:to\s+)?([a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,})\b",
            lowered,
            flags=re.IGNORECASE,
        )
        if email_match and re.search(r"\b(confirm|confirmation|confirming)\b", lowered):
            to_addr = email_match.group(1).strip()
            calendar_fields = _extract_calendar_fields(user_text)
            when_parts: list[str] = []
            if calendar_fields.get("day"):
                when_parts.append(str(calendar_fields["day"]))
            if calendar_fields.get("time"):
                when_parts.append(str(calendar_fields["time"]))
            when_text = " at ".join(when_parts) if len(when_parts) == 2 else (
                when_parts[0] if when_parts else "the scheduled time"
            )
            return (
                f'send an email to {to_addr} say "Confirming our meeting for {when_text}."'
            )
        if re.search(r"\bwhat does\b.*\bemail\b.*\b(say|says)\b", lowered):
            return user_text.strip()
        if re.search(r"\btell me\b.*\bemail\b.*\b(say|says)\b", lowered):
            return user_text.strip()
        if "summarize" in lowered:
            return "check my inbox and summarize the most relevant email thread"
        if re.search(r"\b(reply|draft|send)\b", lowered):
            return f"draft an email reply based on this request: {user_text.strip()}"
        return "check my latest inbox emails"
    if action == "google_drive":
        if re.search(r"\b(attach|relevant|matching)\b", lowered):
            return f"find the most relevant Google Drive file for this request: {user_text.strip()}"
        return "find relevant files in Google Drive"
    if action == "google_calendar":
        if re.search(r"\b(create|schedule|meeting|event|reschedule|move|time|availability)\b", lowered):
            return f"schedule or update a calendar event based on this request: {user_text.strip()}"
        if "remind" in lowered:
            return f"add a reminder event based on this request: {user_text.strip()}"
        return "check my calendar and propose times"
    return user_text.strip()


def _first_keyword_position(text: str, patterns: list[str]) -> int:
    found_positions: list[int] = []
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            found_positions.append(match.start())
    if not found_positions:
        return -1
    return min(found_positions)


def _format_orchestration_response(step_results: list[OrchestrationStepResult]) -> str:
    if not step_results:
        return "No orchestration steps were executed."
    completed = sum(1 for step in step_results if step.success)
    total = len(step_results)
    lines = [
        "Orchestration Summary",
        "---------------------",
        f"Executed {total} step(s). Completed: {completed}.",
        "",
        "Step Results",
        "------------",
    ]
    for index, step in enumerate(step_results, start=1):
        status = "Completed" if step.success else "Failed"
        lines.append(f"{index}. {step.capability_label}: {status}")
        lines.append(f"   Query: {step.query}")
        first_line = (step.assistant_text or "").strip().splitlines()
        if first_line:
            lines.append(f"   Result: {first_line[0]}")
    lines.extend(["", "Detailed Outputs", "----------------"])
    for step in step_results:
        lines.append(f"{step.capability_label}")
        lines.append(step.assistant_text.strip() or "No output.")
        lines.append("")
    return "\n".join(lines).strip()


def _dedupe_sources_by_url(sources: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for source in sources:
        url = (source.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(source)
    return out


def _serialize_orchestration_step_result(step: OrchestrationStepResult) -> dict[str, object]:
    return {
        "action": step.action,
        "tool_name": step.tool_name,
        "query": step.query,
        "success": step.success,
        "reason": step.reason,
        "capability_label": step.capability_label,
        "source_count": len(step.sources),
    }


def _capability_label_for_action(action: str) -> str:
    if action == "google_calendar":
        return "Google Calendar"
    if action == "google_gmail":
        return "Gmail"
    if action == "google_drive":
        return "Google Drive"
    if action == "web_search":
        return "Web Search"
    return action.replace("_", " ").title()


def _should_force_verification_web_search(user_text: str, reasons: list[str]) -> bool:
    # Only force web search for genuinely high-risk factual requests.
    if "high_stakes" not in reasons:
        return False

    text = re.sub(r"\s+", " ", user_text.strip().lower())
    if not text:
        return False

    if any(
        cue in text
        for cue in (
            "write",
            "draft",
            "brainstorm",
            "summarize",
            "rewrite",
            "translate",
            "debug",
            "code",
            "refactor",
        )
    ):
        return False

    return "?" in text or text.startswith(
        ("what", "who", "when", "where", "how", "is", "are", "does", "do", "did", "can")
    )


def _maybe_continue_recent_tool_route(
    text: str,
    previous_action: str | None,
) -> RouteDecision | None:
    if previous_action not in {"google_calendar", "google_gmail", "google_drive"}:
        return None
    lowered = text.strip().lower()
    if not _looks_like_contextual_tool_followup(text):
        return None
    if previous_action == "google_calendar" and re.search(
        r"\b(gmail|email|inbox|thread|draft)\b", lowered
    ):
        return None
    if previous_action == "google_gmail" and re.search(
        r"\b(calendar|schedule|event|meeting)\b", lowered
    ):
        return None
    if previous_action == "google_calendar":
        reason = "calendar_context_followup"
    elif previous_action == "google_gmail":
        reason = "gmail_context_followup"
    else:
        reason = "drive_context_followup"
    return RouteDecision(action=previous_action, reason=reason, confidence=0.9)


def _looks_like_contextual_tool_followup(text: str) -> bool:
    lowered = text.strip().lower()
    if not lowered or len(lowered) > 90:
        return False
    normalized = _normalize_short_reply_text(text)
    if normalized in {
        "check",
        "check now",
        "now check",
        "again",
        "what about",
        "what about today",
        "what about tomorrow",
        "what about this week",
        "what about this month",
        "show me",
        "refresh",
        "now",
    }:
        return True
    if re.search(r"\b(this|next)\s+(week|month)\b", lowered):
        return True
    if re.search(r"\b(today|tomorrow|yesterday)\b", lowered):
        return True
    if re.search(r"\bon\s+\d{1,2}(?:st|nd|rd|th)?\b", lowered):
        return True
    if re.search(r"\bon\s+\w+day\b", lowered):
        return True
    if re.search(r"\b(reschedule|move|shift|update|change)\b", lowered):
        return True
    if re.search(r"\b(that|this|it)\b.*\b(to|for)\b.*\b\d{1,2}(?::\d{2})?\s*(am|pm)\b", lowered):
        return True
    if re.match(
        r"^\s*(ok|okay|alright|cool|great|thanks|thank you|and|so)\b.*\b(check|look|show|again|now)\b",
        lowered,
    ):
        return True
    if re.match(r"^\s*(what|did|how|and)\b", lowered) and re.search(
        r"\b(today|tomorrow|week|month|on)\b", lowered
    ):
        return True
    return False


def _is_calendar_confirmation_required_items(items: list) -> bool:
    if len(items) != 1:
        return False
    title = str(getattr(items[0], "title", "")).strip().lower()
    return title == "confirmation required"


def _is_calendar_created_items(items: list) -> bool:
    for item in items:
        title = str(getattr(item, "title", "")).strip()
        if title.startswith("[Created] "):
            return True
    return False


def _is_calendar_confirmation_reply(text: str) -> bool:
    lowered = text.strip().lower()
    normalized = _normalize_short_reply_text(text)
    if not normalized:
        return False
    if normalized in {
        "confirm",
        "yes",
        "yea",
        "ya",
        "yep",
        "yeah",
        "ok",
        "okay",
        "sure",
        "go ahead",
        "proceed",
        "do it",
        "sounds good",
    }:
        return True
    if normalized.startswith("confirm:") or normalized.startswith("confirm "):
        return True
    if re.search(r"\bi asked you to (add|create|schedule|book|set up|put)\b", lowered):
        return True
    if re.match(r"^\s*(please\s+)?(add|create|schedule|book|set up|put)\b", lowered):
        return True
    return bool(
        re.match(
            r"^\s*(yes|yep|yeah|ok|okay|go ahead|proceed|do it|sounds good)\b",
            lowered,
        )
    )


def _is_standalone_calendar_confirmation_reply(text: str) -> bool:
    normalized = _normalize_short_reply_text(text)
    return normalized in {
        "confirm",
        "yes",
        "yea",
        "ya",
        "yep",
        "yeah",
        "ok",
        "okay",
        "sure",
        "go ahead",
        "proceed",
        "do it",
        "sounds good",
        "add it",
        "add it to calendar",
    }


def _is_calendar_cancel_reply(text: str) -> bool:
    lowered = text.strip().lower()
    if not lowered:
        return False
    return bool(re.match(r"^\s*(no|cancel|stop|don't|do not)\b", lowered))


def _is_calendar_pause_reply(text: str) -> bool:
    lowered = text.strip().lower()
    if not lowered:
        return False
    return bool(re.match(r"^\s*(wait|hold|not now|later|pause)\b", lowered))


def _is_calendar_draft_edit_reply(text: str) -> bool:
    lowered = text.strip().lower()
    if not lowered:
        return False
    if _is_calendar_confirmation_reply(text) or _is_calendar_cancel_reply(text):
        return False

    if re.search(
        r"\b(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        lowered,
    ):
        return True
    if re.search(
        r"\b(?:"
        r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
        r")\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b"
        r"|"
        r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b",
        lowered,
    ):
        return True
    if re.search(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b", lowered):
        return True
    if re.search(
        r"\b(change|update|instead|not|move|shift|title|location|zoom|meet\.?|teams?)\b",
        lowered,
    ):
        return True
    return False


def _build_updated_calendar_draft_request(draft_text: str, followup_text: str) -> str:
    draft_fields = _extract_calendar_fields(draft_text)
    followup_fields = _extract_calendar_fields(followup_text)
    merged = {
        "title": followup_fields.get("title") or draft_fields.get("title"),
        "day": followup_fields.get("day") or draft_fields.get("day"),
        "time": followup_fields.get("time") or draft_fields.get("time"),
        "location": followup_fields.get("location") or draft_fields.get("location"),
    }
    parts: list[str] = []
    if merged["title"]:
        parts.append(str(merged["title"]))
    if merged["day"]:
        parts.append(f"on {merged['day']}")
    if merged["time"]:
        parts.append(f"at {merged['time']}")
    if merged["location"]:
        parts.append(f"in {merged['location']}")
    if not parts:
        return draft_text.strip()
    return "add event " + " ".join(parts)


def _is_gmail_send_confirmation_reply(text: str) -> bool:
    lowered = text.strip().lower()
    if not lowered:
        return False
    if lowered == "confirm":
        return True
    return bool(
        lowered.startswith("confirm send")
        or lowered.startswith("confirm: send")
        or "yes, send draft" in lowered
        or "yes send draft" in lowered
        or re.match(r"^\s*(yes|ok|okay|go ahead|proceed)\b", lowered)
    )


def _is_meta_web_search_request(text: str) -> bool:
    lowered = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not lowered:
        return False
    patterns = (
        r"^can(?:not|'t)? you search the web\??$",
        r"^could you search the web\??$",
        r"^can you look it up\??$",
        r"^please search the web\??$",
        r"^search the web\??$",
        r"^look this up\??$",
    )
    return any(re.match(pattern, lowered) for pattern in patterns)


def _is_generic_web_followup_request(text: str) -> bool:
    lowered = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not lowered:
        return False
    patterns = (
        r"^search the web for more options\??$",
        r"^search the web for more\??$",
        r"^search the web for more results\??$",
        r"^search for more options\??$",
        r"^find more options\??$",
        r"^show me more options\??$",
        r"^more options\??$",
    )
    return any(re.match(pattern, lowered) for pattern in patterns)


def _build_followup_web_query(followup_text: str, previous_query: str) -> str:
    lowered = re.sub(r"\s+", " ", (followup_text or "").strip().lower())
    if "few options" in lowered or "some options" in lowered:
        return f"{previous_query.strip()} more options"
    target = _extract_link_followup_target(lowered)
    if target:
        if _looks_like_specific_product_target(target) or _looks_like_purchase_intent_query(previous_query):
            return f"{target} official product page buy"
        return f"{target} official page"
    if _is_link_only_web_followup_request(lowered):
        return previous_query.strip()
    if "more options" in lowered:
        return f"{previous_query.strip()} more options"
    if "more results" in lowered or lowered.startswith("more"):
        return f"{previous_query.strip()} more results"
    return previous_query.strip()


def _is_link_only_web_followup_request(text: str) -> bool:
    lowered = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not lowered:
        return False
    patterns = (
        r"^can you link me\??$",
        r"^could you link me\??$",
        r"^link me\??$",
        r"^send me links\??$",
        r"^give me links\??$",
        r"^show links\??$",
        r"^can you send me the links\??$",
        r"^can you send links\??$",
    )
    return any(re.match(pattern, lowered) for pattern in patterns)


def _is_link_followup_request(text: str) -> bool:
    return _is_link_only_web_followup_request(text) or bool(_extract_link_followup_target(text))


def _extract_link_followup_target(text: str) -> str:
    lowered = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not lowered:
        return ""
    match = re.search(
        r"\b(?:can you|could you|please)?\s*(?:send me|give me|show me|link me)?\s*links?\s*(?:to|for)\s+(?:the|teh|a|an)?\s*(.+)$",
        lowered,
    )
    if not match:
        match = re.search(
            r"\b(?:can you|could you|please)?\s*link me\s*(?:to|for)\s+(?:the|teh|a|an)?\s*(.+)$",
            lowered,
        )
    if not match:
        return ""
    target = re.sub(r"[?.!,]+$", "", match.group(1).strip())
    # Drop long spec tails so query stays focused on the product model.
    target = re.split(r"[:,]", target, maxsplit=1)[0].strip()
    target = re.sub(r"\b(that|this)\b", "", target).strip()
    target = re.sub(r"\b(the|a|an)\b", " ", target).strip()
    target = re.sub(r"\b(laptop|notebook|computer)\b", " ", target).strip()
    target = re.sub(r"\s+", " ", target).strip()
    target = re.sub(r"\b(\w+)\s+\1\b", r"\1", target)
    if target in {"few options", "some options", "options", "more options", "few", "some"}:
        return ""
    return target


def _looks_like_specific_product_target(target: str) -> bool:
    lowered = target.lower().strip()
    if not lowered:
        return False
    model_keywords = (
        "xps",
        "thinkpad",
        "macbook",
        "zephyrus",
        "predator",
        "omen",
        "blade",
        "legion",
        "inspiron",
        "vivobook",
        "zenbook",
        "rog",
    )
    has_model_keyword = any(keyword in lowered for keyword in model_keywords)
    has_model_number = bool(re.search(r"\b[a-z]{1,5}\s*\d{2,4}[a-z]?\b|\b\d{2,4}\b", lowered))
    has_brand = bool(
        re.search(
            r"\b(dell|lenovo|apple|asus|msi|acer|hp|razer|alienware|samsung)\b",
            lowered,
        )
    )
    return has_model_keyword or (has_brand and has_model_number)


def _looks_like_purchase_intent_query(query: str) -> bool:
    lowered = (query or "").strip().lower()
    if not lowered:
        return False
    return bool(
        re.search(
            r"\b(buy|purchase|price|pricing|budget|under \$?\d+|best|gaming|laptop|boat|tractor|car|model|option)s?\b",
            lowered,
        )
    )


def _is_gmail_cancel_reply(text: str) -> bool:
    lowered = text.strip().lower()
    if not lowered:
        return False
    return bool(re.match(r"^\s*(no|cancel|stop|don't|do not)\b", lowered))


def _build_confirmed_gmail_send_request(pending_text: str, followup_text: str) -> str:
    followup = followup_text.strip()
    lowered = followup.lower()
    if lowered.startswith("confirm send") or lowered.startswith("confirm: send"):
        return followup
    draft_id = _extract_draft_id_for_confirmation(pending_text)
    if not draft_id:
        draft_id = _extract_draft_id_for_confirmation(followup_text)
    if not draft_id:
        return "confirm send draft"
    return f"confirm send draft {draft_id}"


def _extract_draft_id_for_confirmation(text: str) -> str | None:
    match = re.search(
        r"\bdraft(?:\s+id)?\s*[:#]?\s*([a-z0-9_-]{3,})\b",
        text,
        re.IGNORECASE,
    )
    if not match:
        compose_match = re.search(r"\bcompose=([a-z0-9_-]{3,})\b", text, re.IGNORECASE)
        if not compose_match:
            return None
        return compose_match.group(1).strip()
    return match.group(1).strip()


def _is_gmail_send_confirmation_required_items(items: list) -> bool:
    if len(items) != 1:
        return False
    title = str(getattr(items[0], "title", "")).strip().lower()
    return title == "send confirmation required"


def _is_gmail_sent_items(items: list) -> bool:
    for item in items:
        title = str(getattr(item, "title", "")).strip().lower()
        if title.startswith("[sent]"):
            return True
    return False


def _extract_gmail_send_pending_text(items: list, fallback: str) -> str:
    if len(items) == 1:
        title = str(getattr(items[0], "title", "")).strip().lower()
        if title == "send confirmation required":
            snippet = str(getattr(items[0], "snippet", "")).strip()
            item_url = str(getattr(items[0], "url", "")).strip()
            draft_id = _extract_draft_id_for_confirmation(snippet)
            if not draft_id:
                draft_id = _extract_draft_id_for_confirmation(item_url)
            if draft_id:
                return f"confirm send draft {draft_id}"
            if snippet:
                return snippet
    return fallback


def _build_confirmed_calendar_request(draft_text: str, followup_text: str) -> str:
    followup = followup_text.strip()
    lowered = followup.lower()
    if lowered.startswith("confirm:"):
        payload = followup.split(":", 1)[1].strip()
        if payload:
            return followup
    elif lowered.startswith("confirm "):
        payload = followup[len("confirm ") :].strip()
        if payload:
            return followup

    draft_fields = _extract_calendar_fields(draft_text)
    followup_fields = _extract_calendar_fields(followup_text)
    merged = {
        "title": followup_fields.get("title") or draft_fields.get("title"),
        "day": followup_fields.get("day") or draft_fields.get("day"),
        "time": followup_fields.get("time") or draft_fields.get("time"),
        "location": followup_fields.get("location") or draft_fields.get("location"),
    }
    parts: list[str] = []
    if merged["title"]:
        parts.append(str(merged["title"]))
    if merged["day"]:
        parts.append(f"on {merged['day']}")
    if merged["time"]:
        parts.append(f"at {merged['time']}")
    if merged["location"]:
        parts.append(f"in {merged['location']}")
    if not parts:
        return f"confirm: {draft_text.strip()}"
    return "confirm: add event " + " ".join(parts)


def _extract_calendar_fields(text: str) -> dict[str, str]:
    raw = text.strip()
    lowered = text.strip().lower()
    out: dict[str, str] = {}
    title_line = re.search(r"(?im)^\s*-\s*title:\s*(.+?)\s*$", raw)
    if title_line:
        out["title"] = title_line.group(1).strip()
    day_line = re.search(r"(?im)^\s*-\s*day:\s*(.+?)\s*$", raw)
    if day_line:
        out["day"] = day_line.group(1).strip()
    time_line = re.search(r"(?im)^\s*-\s*time:\s*(.+?)\s*$", raw)
    if time_line:
        out["time"] = time_line.group(1).replace(" ", "").strip()
    location_line = re.search(r"(?im)^\s*-\s*location:\s*(.+?)\s*$", raw)
    if location_line:
        out["location"] = location_line.group(1).strip()

    if "title" not in out:
        title_override = re.search(
            r"\bchange\s+(?:the\s+)?title\s+to\s+(.+?)(?=(?:\s+(?:on|at|in)\b)|$)",
            lowered,
        )
        if title_override:
            out["title"] = _title_case_words(title_override.group(1))
        else:
            with_match = re.search(
                r"\bwith\s+([a-z][a-z\s'-]{1,50}?)(?=\s+(?:on|at|in|and|for|its|it'?s|please|add|create|schedule|book)\b|[,.!?]|$)",
                lowered,
            )
            if with_match:
                out["title"] = "Meeting with " + _title_case_words(with_match.group(1))

    if "day" not in out:
        day_match = re.search(
            r"\b(today|tonight|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            lowered,
        )
        if day_match:
            token = day_match.group(1)
            out["day"] = "today" if token == "tonight" else token
        else:
            date_match = re.search(
                r"\b(?:"
                r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
                r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
                r")\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b"
                r"|"
                r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b",
                lowered,
            )
            if date_match:
                out["day"] = date_match.group(0)

    if "time" not in out:
        time_match = re.search(r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm))\b", lowered)
        if time_match:
            out["time"] = time_match.group(1).replace(" ", "")

    if "location" not in out:
        location_match = re.search(
            r"\bin\s+([a-z][a-z\s,.'-]{1,60}?)(?=(?:\s+(?:on|at)\b)|$)",
            lowered,
        )
        if location_match:
            out["location"] = _title_case_words(location_match.group(1))
        else:
            at_location_match = re.search(
                r"\bat\s+([a-z][a-z\s,.'-]{1,60}?)(?=(?:\s+(?:on|in)\b)|$)",
                lowered,
            )
            if at_location_match:
                out["location"] = _title_case_words(at_location_match.group(1))
    return out


def _title_case_words(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value.strip(" ."))
    return " ".join(part.capitalize() for part in cleaned.split(" ") if part)


def _normalize_short_reply_text(text: str) -> str:
    lowered = (text or "").strip().lower()
    if not lowered:
        return ""
    cleaned = re.sub(r"[.!?]+$", "", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def _extract_calendar_draft_text(items: list, fallback: str) -> str:
    if len(items) == 1:
        title = str(getattr(items[0], "title", "")).strip().lower()
        if title == "confirmation required":
            snippet = str(getattr(items[0], "snippet", "")).strip()
            if snippet:
                return snippet
    return fallback


def _prevent_unexecuted_action_claims(assistant_text: str, routed_action: str) -> str:
    if routed_action in {"google_calendar", "google_gmail", "google_drive"}:
        return assistant_text
    lowered = (assistant_text or "").strip().lower()
    if not lowered:
        return assistant_text
    calendar_claim = "calendar" in lowered and bool(
        re.search(
            r"\b(i(?:'ve| have)?\s+(added|scheduled|booked|put)|it(?:'s| is)\s+added|added your)\b",
            lowered,
        )
    )
    gmail_claim = bool(
        re.search(
            r"\b(i(?:'ve| have)?\s+(sent|emailed)|email sent|sent successfully|i did send)\b",
            lowered,
        )
    )
    if calendar_claim:
        return (
            "I haven’t added anything to your Google Calendar yet. "
            "I can do that through the calendar tool now if you want."
        )
    if gmail_claim:
        return (
            "I haven’t sent an email yet. "
            "I can draft it and send only after your confirmation."
        )
    return assistant_text
