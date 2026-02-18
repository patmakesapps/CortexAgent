from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
import json
from uuid import uuid4
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib import request as urlrequest

from cortexagent.config import settings
from cortexagent.models import AgentDecision
from cortexagent.router.intent_router import decide_action
from cortexagent.services.connected_accounts_repo import (
    ConnectedAccountsRepository,
    ConnectedAccountUpsert,
)
from cortexagent.services.cortexltm_client import CortexLTMClient
from cortexagent.services.confirmation_intent import (
    classify_pending_confirmation_intent,
)
from cortexagent.services.google_oauth import GoogleOAuthService
from cortexagent.services.planner import build_orchestration_plan
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
    expected_outcome: str
    step_id: str = ""
    operation: str = ""
    args: dict[str, object] | None = None
    requires_confirmation: bool = False
    depends_on: list[str] | None = None
    why: str = ""


@dataclass(frozen=True)
class PendingAction:
    pending_action_id: str
    thread_id: str
    action: str
    operation: str
    args: dict[str, object]
    query: str
    expected_outcome: str
    created_at: str
    status: str


@dataclass(frozen=True)
class OrchestrationStepResult:
    action: str
    tool_name: str
    query: str
    assistant_text: str
    sources: list[dict[str, str]]
    success: bool
    reason: str
    execution_status: str
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
        self._pending_actions_by_thread: dict[str, list[PendingAction]] = {}
        self._last_tool_action_by_thread: dict[str, str] = {}
        self._last_non_meta_user_text_by_thread: dict[str, str] = {}
        self._last_web_search_query_by_thread: dict[str, str] = {}
        self._last_pipeline_results_by_thread: dict[str, list[dict[str, object]]] = {}
        self._last_cleared_pending_count_by_thread: dict[str, int] = {}

    def handle_chat(
        self,
        thread_id: str,
        text: str,
        short_term_limit: int | None,
        authorization: str | None,
    ) -> OrchestratorResult:
        decision_mode = settings.agent_decision_mode
        if decision_mode not in {"hybrid", "llm_first", "llm_only"}:
            decision_mode = "llm_only"
        llm_only_mode = True
        previous_non_meta_user_text = self._last_non_meta_user_text_by_thread.get(thread_id)
        is_meta_web_request = False
        if not is_meta_web_request:
            self._last_non_meta_user_text_by_thread[thread_id] = text

        pending_actions = list(self._pending_actions_by_thread.get(thread_id, []))
        if pending_actions:
            pending_action_id = _extract_pending_action_id(text)
            pending_calendar = any(action.action == "google_calendar" for action in pending_actions)
            pending_gmail = any(action.action == "google_gmail" for action in pending_actions)
            pending_intent = classify_pending_confirmation_intent(
                text=text,
                pending_calendar=pending_calendar,
                pending_gmail=pending_gmail,
            )
            selected_actions = _select_pending_actions(
                pending_actions=pending_actions,
                pending_action_id=pending_action_id,
            )
            selected_ids = {action.pending_action_id for action in selected_actions}
            if _looks_like_pending_status_query(text):
                return OrchestratorResult(
                    response=_render_pending_actions_status(pending_actions=pending_actions),
                    decision=AgentDecision(
                        action="orchestration",
                        reason="pending_action_status",
                        confidence=1.0,
                    ),
                    sources=[],
                )
            if pending_intent.intent == "cancel":
                self._pending_actions_by_thread[thread_id] = [
                    action for action in pending_actions if action.pending_action_id not in selected_ids
                ]
                if selected_ids:
                    self._last_cleared_pending_count_by_thread[thread_id] = len(selected_ids)
                return OrchestratorResult(
                    response="Canceled the pending action request.",
                    decision=AgentDecision(
                        action="orchestration",
                        reason="pending_action_canceled",
                        confidence=1.0,
                    ),
                    sources=[],
                )
            if pending_intent.intent == "edit":
                has_calendar_selection = any(
                    action.action == "google_calendar"
                    and action.pending_action_id in selected_ids
                    for action in pending_actions
                )
                updated_pending_actions = _apply_pending_action_edits(
                    pending_actions=pending_actions,
                    selected_ids=selected_ids,
                    followup_text=text,
                )
                self._pending_actions_by_thread[thread_id] = updated_pending_actions
                return OrchestratorResult(
                    response=(
                        (
                            "Updated your pending calendar details. The action is still pending confirmation."
                            if has_calendar_selection
                            else "I still have pending actions waiting for confirmation."
                        )
                        + "\n\n"
                        + _render_pending_actions_message(updated_pending_actions)
                    ),
                    decision=AgentDecision(
                        action="orchestration",
                        reason="pending_action_edited",
                        confidence=1.0,
                    ),
                    sources=[],
                )
            if pending_intent.intent == "confirm":
                confirmed_steps = [
                    _build_confirmed_step_from_pending(action=action, followup_text=text)
                    for action in selected_actions
                ]
                self._pending_actions_by_thread[thread_id] = [
                    action for action in pending_actions if action.pending_action_id not in selected_ids
                ]
                if selected_ids:
                    self._last_cleared_pending_count_by_thread[thread_id] = len(selected_ids)
                return self._run_multi_step_pipeline(
                    thread_id=thread_id,
                    text=text,
                    authorization=authorization,
                    steps=confirmed_steps,
                    planner_meta={
                        "planner_used": False,
                        "planner_attempted": False,
                        "planner_source": "pending_confirmation",
                        "validation_result": "pending_confirmation",
                        "fallback_reason": None,
                    },
                )
            if not _looks_like_new_request_while_pending(
                text=text,
                pending_intent=pending_intent.intent,
            ):
                return OrchestratorResult(
                    response=(
                        "I still have pending actions waiting for confirmation.\n\n"
                        + _render_pending_actions_message(pending_actions)
                    ),
                    decision=AgentDecision(
                        action="orchestration",
                        reason="pending_action_waiting_confirmation",
                        confidence=1.0,
                    ),
                    sources=[],
                )
        if _looks_like_pending_status_query(text):
            last_count = self._last_cleared_pending_count_by_thread.get(thread_id)
            return OrchestratorResult(
                response=_render_no_pending_status(
                    text=text,
                    last_cleared_count=last_count,
                ),
                decision=AgentDecision(
                    action="orchestration",
                    reason="pending_action_status_no_pending",
                    confidence=1.0,
                ),
                sources=[],
            )

        planner_meta: dict[str, object] | None = None
        if settings.agent_tools_enabled:
            planner_decision = build_orchestration_plan(
                user_text=text,
                prior_user_text=previous_non_meta_user_text,
                prior_tool_action=self._last_tool_action_by_thread.get(thread_id),
                minimum_tool_steps=1,
                tool_intent_required=False,
                tool_intent_hints=[],
            )
            planner_meta = dict(planner_decision.metadata)
            planner_meta["decision_mode"] = decision_mode
            if planner_decision.steps:
                planned_steps = [
                    OrchestrationStep(
                        step_id=step.step_id,
                        action=step.action,
                        operation=step.operation,
                        args=step.args,
                        query=step.query,
                        expected_outcome=step.expected_outcome,
                        requires_confirmation=step.requires_confirmation,
                        depends_on=step.depends_on,
                        why=step.why,
                    )
                    for step in planner_decision.steps
                ]
                return self._run_multi_step_pipeline(
                    thread_id=thread_id,
                    text=text,
                    authorization=authorization,
                    steps=planned_steps,
                    planner_meta=planner_meta,
                )
            fallback_reason = str(planner_meta.get("fallback_reason") or "").strip().lower()
            deterministic_steps = _build_multi_step_plan(text)
            if deterministic_steps:
                return self._run_multi_step_pipeline(
                    thread_id=thread_id,
                    text=text,
                    authorization=authorization,
                    steps=deterministic_steps,
                    planner_meta={
                        **planner_meta,
                        "planner_source": "deterministic_fallback_plan",
                        "validation_result": "deterministic_fallback_plan",
                    },
                )

            route_decision = decide_action(
                user_text=text,
                tools_enabled=settings.agent_tools_enabled,
                web_search_enabled=settings.web_search_enabled,
                prior_user_text=previous_non_meta_user_text,
                prior_tool_action=self._last_tool_action_by_thread.get(thread_id),
            )
            routed_step = _build_routed_fallback_step(
                route_action=route_decision.action,
                text=text,
            )
            if routed_step is not None:
                return self._run_multi_step_pipeline(
                    thread_id=thread_id,
                    text=text,
                    authorization=authorization,
                    steps=[routed_step],
                    planner_meta={
                        **planner_meta,
                        "planner_source": "router_intent_fallback",
                        "validation_result": "router_intent_fallback",
                        "router_reason": route_decision.reason,
                        "router_confidence": route_decision.confidence,
                    },
                )

            if _looks_like_tool_access_request(text):
                explicit_step = _infer_explicit_tool_step(
                    text=text,
                    prior_tool_action=self._last_tool_action_by_thread.get(thread_id),
                )
                if explicit_step is not None:
                    return self._run_multi_step_pipeline(
                        thread_id=thread_id,
                        text=text,
                        authorization=authorization,
                        steps=[explicit_step],
                        planner_meta={
                            **planner_meta,
                            "planner_source": "intent_fallback",
                            "validation_result": "intent_fallback",
                        },
                    )
                if fallback_reason in {
                    "missing_api_key",
                    "missing_model",
                    "unsupported_provider",
                    "request_failed",
                    "network_error",
                    "http_error",
                    "timeout",
                }:
                    return OrchestratorResult(
                        response=(
                            "Tool planning is currently unavailable, so I cannot execute that tool action right now. "
                            "Please check planner/router LLM provider configuration and API keys."
                        ),
                        decision=AgentDecision(
                            action="chat",
                            reason=f"tool_planning_unavailable:{fallback_reason or 'unknown'}",
                            confidence=1.0,
                        ),
                        sources=[],
                    )
                if fallback_reason in {"invalid_steps", "invalid_schema", "low_confidence"}:
                    return OrchestratorResult(
                        response=(
                            "I could not produce a valid tool plan for that request. "
                            "Please rephrase with one concrete action, for example: "
                            "'list my latest Gmail threads' or 'show my calendar events today'."
                        ),
                        decision=AgentDecision(
                            action="chat",
                            reason=f"tool_plan_invalid:{fallback_reason}",
                            confidence=1.0,
                        ),
                        sources=[],
                    )

        verification = assess_verification_profile(text)
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
            routed_action="chat",
            user_text=text,
        )
        return OrchestratorResult(
            response=assistant_text,
            decision=AgentDecision(
                action="chat",
                reason="planner_fallback_chat",
                confidence=0.6,
            ),
            sources=[],
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
        planner_meta: dict[str, object] | None = None,
    ) -> OrchestratorResult:
        step_results: list[OrchestrationStepResult] = []
        for step in steps:
            if step.requires_confirmation and _is_write_step(step):
                pending = _make_pending_action(thread_id=thread_id, step=step)
                thread_pending = list(self._pending_actions_by_thread.get(thread_id, []))
                thread_pending.append(pending)
                self._pending_actions_by_thread[thread_id] = thread_pending
                step_results.append(
                    OrchestrationStepResult(
                        action=step.action,
                        tool_name=step.action,
                        query=step.query,
                        assistant_text=_render_pending_action_created_message(pending),
                        sources=[],
                        success=False,
                        reason="confirmation_required",
                        execution_status="action_required",
                        capability_label=_capability_label_for_action(step.action),
                        extra_meta={"pending_action_id": pending.pending_action_id},
                    )
                )
                continue
            result = self._execute_pipeline_step(
                thread_id=thread_id,
                user_text=text,
                step=step,
                authorization=authorization,
            )
            step_results.append(result)
            pending_from_result = _pending_action_from_step_result(
                thread_id=thread_id,
                step=step,
                result=result,
            )
            if pending_from_result is not None:
                thread_pending = list(self._pending_actions_by_thread.get(thread_id, []))
                thread_pending.append(pending_from_result)
                self._pending_actions_by_thread[thread_id] = thread_pending
            if (
                result.execution_status in {"completed", "action_required"}
                and result.action in {"google_calendar", "google_gmail", "google_drive"}
            ):
                self._last_tool_action_by_thread[thread_id] = result.action
            if result.execution_status == "completed" and result.action == "web_search":
                self._last_web_search_query_by_thread[thread_id] = result.query

        combined_sources = _dedupe_sources_by_url(
            [source for step in step_results for source in step.sources]
        )
        response = _format_orchestration_response(step_results=step_results)
        planner_source = str((planner_meta or {}).get("planner_source") or "").strip().lower()
        decision_reason = "multi_step_pipeline_executed"
        if planner_source == "llm":
            decision_reason = "multi_step_pipeline_executed_llm_planner"
        elif planner_source:
            decision_reason = f"multi_step_pipeline_executed_{planner_source}"
        decision = AgentDecision(
            action="orchestration",
            reason=decision_reason,
            confidence=0.93,
        )
        self._persist_orchestration_events(
            thread_id=thread_id,
            user_text=text,
            assistant_text=response,
            authorization=authorization,
            step_results=step_results,
            sources=combined_sources,
            planner_meta=planner_meta,
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
        expected_outcome = step.expected_outcome
        step_operation = (step.operation or "").strip().lower()
        step_args = dict(step.args or {})
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
                    execution_status="failed",
                    capability_label=capability_label,
                )
            try:
                tool = self.tool_registry.get(tool_name)
                result = tool.run(
                    ToolContext(
                        thread_id=thread_id,
                        user_text=query,
                        tool_meta={
                            "access_token": access_token,
                            "max_results": 8,
                            "operation": step_operation,
                            "args": step_args,
                        },
                    )
                )
                assistant_text, sources = _format_google_calendar_response(result.items)
                assistant_text = _polish_response_text(assistant_text)
                execution_status, reason = _evaluate_calendar_step_execution(
                    expected_outcome=expected_outcome,
                    items=result.items,
                )
                assistant_text = _annotate_orchestration_step_text(
                    action=action,
                    expected_outcome=expected_outcome,
                    execution_status=execution_status,
                    assistant_text=assistant_text,
                )
                return OrchestrationStepResult(
                    action=action,
                    tool_name=result.tool_name,
                    query=result.query,
                    assistant_text=assistant_text,
                    sources=sources,
                    success=execution_status == "completed",
                    reason=reason,
                    execution_status=execution_status,
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
                    execution_status="failed",
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
                    execution_status="failed",
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
                            "operation": step_operation,
                            "args": step_args,
                        },
                    )
                )
                assistant_text, sources = _format_google_gmail_response(result.items)
                assistant_text = _polish_response_text(assistant_text)
                execution_status, reason = _evaluate_gmail_step_execution(
                    expected_outcome=expected_outcome,
                    items=result.items,
                )
                assistant_text = _annotate_orchestration_step_text(
                    action=action,
                    expected_outcome=expected_outcome,
                    execution_status=execution_status,
                    assistant_text=assistant_text,
                )
                return OrchestrationStepResult(
                    action=action,
                    tool_name=result.tool_name,
                    query=result.query,
                    assistant_text=assistant_text,
                    sources=sources,
                    success=execution_status == "completed",
                    reason=reason,
                    execution_status=execution_status,
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
                    execution_status="failed",
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
                    execution_status="failed",
                    capability_label=capability_label,
                )
            try:
                tool = self.tool_registry.get(tool_name)
                result = tool.run(
                    ToolContext(
                        thread_id=thread_id,
                        user_text=query,
                        tool_meta={
                            "access_token": access_token,
                            "max_results": 8,
                            "operation": step_operation,
                            "args": step_args,
                        },
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
                    execution_status="completed",
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
                    execution_status="failed",
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
                    execution_status="completed",
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
                    execution_status="failed",
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
            execution_status="failed",
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
        planner_meta: dict[str, object] | None = None,
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
                            "execution_status": step.execution_status,
                        }
                        for step in step_results
                    ],
                    "planner": planner_meta or {},
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
                    "steps": [
                        {
                            "action": step.action,
                            "toolName": step.tool_name,
                            "success": step.success,
                            "reason": step.reason,
                            "executionStatus": step.execution_status,
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
    display_items = items
    if has_created:
        created_items = [
            item
            for item in items
            if isinstance(getattr(item, "title", None), str)
            and str(getattr(item, "title", "")).startswith("[Created] ")
        ]
        if created_items:
            display_items = created_items

    lines = ["Google Calendar updated."] if has_created else ["Upcoming Google Calendar events:"]
    sources: list[dict[str, str]] = []
    for idx, item in enumerate(display_items[:8], start=1):
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
    # Repair malformed nested markdown links like [[label](http://...))
    out = re.sub(r"\[\[([^\]]+)\]\((https?://[^)\s]+)\)\)", r"[\1](\2)", out)
    # Repair dangling extra ')' in normal markdown links.
    out = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)\)", r"[\1](\2)", out)
    out = re.sub(r"\*\*(.*?)\*\*", r"\1", out)
    out = re.sub(r"__([^_]+)__", r"\1", out)
    out = re.sub(r"(^|\n)(\d+)\.([A-Za-z])", r"\1\2. \3", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _looks_like_tool_access_request(text: str) -> bool:
    return _infer_explicit_tool_step(text=text, prior_tool_action=None) is not None


def _infer_explicit_tool_step(text: str, prior_tool_action: str | None = None) -> OrchestrationStep | None:
    lowered = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not lowered:
        return None
    if prior_tool_action in {"google_calendar", "google_gmail", "google_drive", "web_search"}:
        if _looks_like_context_followup_request(lowered):
            contextual = _build_routed_fallback_step(
                route_action=prior_tool_action,
                text=text.strip(),
                why="context_followup_fallback",
            )
            if contextual is not None:
                return contextual
    if _looks_like_gmail_access_intent(lowered):
        return _build_routed_fallback_step(
            route_action="google_gmail",
            text=text.strip(),
            why="explicit_intent_fallback",
        )
    if _looks_like_calendar_access_intent(lowered):
        return _build_routed_fallback_step(
            route_action="google_calendar",
            text=text.strip(),
            why="explicit_intent_fallback",
        )
    if _looks_like_drive_access_intent(lowered):
        return _build_routed_fallback_step(
            route_action="google_drive",
            text=text.strip(),
            why="explicit_intent_fallback",
        )
    if _looks_like_web_search_access_intent(lowered):
        return _build_routed_fallback_step(
            route_action="web_search",
            text=text.strip(),
            why="explicit_intent_fallback",
        )
    return None


def _looks_like_context_followup_request(text: str) -> bool:
    if not text:
        return False
    return bool(
        re.search(
            r"\b(check|show|list|open|what|how|did|status|again|now|still)\b",
            text,
        )
    )


def _looks_like_gmail_access_intent(text: str) -> bool:
    if not text:
        return False
    has_email_target = bool(
        re.search(r"\b(gmail|inbox|email|thread|message|draft|reply)\b", text)
        or re.search(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}", text, flags=re.IGNORECASE)
    )
    has_email_verb = bool(
        re.search(r"\b(check|show|list|read|open|find|search|summarize|draft|reply|send|email)\b", text)
    )
    return has_email_target and has_email_verb


def _looks_like_calendar_access_intent(text: str) -> bool:
    if not text:
        return False
    if _looks_like_calendar_write_request(text):
        return True
    has_calendar_target = bool(
        re.search(r"\b(calendar|events?|meeting|appointment|availability|schedule)\b", text)
    )
    has_calendar_verb = bool(
        re.search(r"\b(check|show|list|view|open|what|when|upcoming|next)\b", text)
    )
    return has_calendar_target and has_calendar_verb


def _looks_like_drive_access_intent(text: str) -> bool:
    if not text:
        return False
    if re.search(r"\bgoogle drive\b", text):
        return True
    if re.search(r"\bmy drive\b", text):
        return True
    if re.search(r"\bdrive\s+(?:file|files|doc|docs|document|documents|folder|folders)\b", text):
        return True
    has_drive_target = bool(re.search(r"\b(file|files|doc|docs|document|documents|folder|folders)\b", text))
    has_drive_verb = bool(re.search(r"\b(check|show|list|find|search|open|look)\b", text))
    has_drive_context = bool(re.search(r"\bdrive\b", text))
    return has_drive_target and has_drive_verb and has_drive_context


def _looks_like_web_search_access_intent(text: str) -> bool:
    if not text:
        return False
    return bool(
        re.search(r"\b(search the web|web search|look up|lookup|find online|search online)\b", text)
    )


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

    expected_outcomes: dict[str, str] = {}
    if gmail_requested:
        expected_outcomes["google_gmail"] = (
            "gmail_send" if _looks_like_gmail_send_request(lowered) else "gmail_read"
        )
    if drive_requested:
        expected_outcomes["google_drive"] = "drive_search"
    if calendar_requested:
        expected_outcomes["google_calendar"] = (
            "calendar_write"
            if _looks_like_calendar_write_request(lowered)
            else "calendar_read"
        )
    if web_requested:
        expected_outcomes["web_search"] = "web_search"

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
        expected_outcome = expected_outcomes.get(action, "generic")
        operation = "read"
        if action == "google_gmail":
            operation = "send" if expected_outcome == "gmail_send" else "read"
        elif action == "google_calendar":
            operation = "create" if expected_outcome == "calendar_write" else "read"
        elif action == "google_drive":
            operation = "search"
        elif action == "web_search":
            operation = "search"
        query = _build_step_query(
            action=action,
            user_text=text,
            expected_outcome=expected_outcome,
        )
        steps.append(
            OrchestrationStep(
                step_id=f"fallback_step_{len(steps) + 1}",
                action=action,
                operation=operation,
                args={},
                query=query,
                expected_outcome=expected_outcome,
                requires_confirmation=expected_outcome in {"gmail_send", "calendar_write"},
                depends_on=[],
                why="deterministic_fallback_plan",
            )
        )
    return steps


def _looks_like_gmail_send_request(text: str) -> bool:
    if not text:
        return False
    has_email_address = bool(
        re.search(
            r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}",
            text,
            flags=re.IGNORECASE,
        )
    )
    explicit_read_request = bool(
        re.search(
            r"\b(read|check|list|show|open)\b.*\b(email|gmail|inbox|thread|message)\b",
            text,
        )
    )
    if explicit_read_request and not has_email_address:
        return False
    if re.search(r"\b(confirm\s+send|send\s+draft|yes\s+send)\b", text):
        return True
    if has_email_address and re.search(r"\b(send|sned|snd|email|compose|write|tell)\b", text):
        return True
    return bool(
        re.search(
            r"\bemail\s+(?:to\s+)?[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _looks_like_calendar_write_request(text: str) -> bool:
    if not text:
        return False
    if re.search(r"\badd\s+it\s+to\s+my\s+calendar\b", text):
        return True
    explicit_read_query = bool(
        re.search(r"\b(check|show|list|view|see|what(?:'s| is)|when)\b", text)
        and re.search(r"\b(calendar|schedule|events?|meetings?)\b", text)
    )
    explicit_write_query = bool(
        re.search(
            r"\b(add|ad|create|schedule|book|set up|put|reschedule|move|shift|update|change|remind)\b",
            text,
        )
    )
    if explicit_read_query and not explicit_write_query:
        return False
    has_write_verb = bool(
        re.search(
            r"\b(add|ad|create|schedule|book|set up|put|reschedule|move|shift|update|change|remind)\b",
            text,
        )
    )
    has_calendar_or_event_ref = bool(
        re.search(r"\b(calendar|meeting|event|appointment|call|reminder)\b", text)
    )
    has_calendar_pronoun = bool(
        "calendar" in text and re.search(r"\b(it|this|that)\b", text)
    )
    return has_write_verb and (has_calendar_or_event_ref or has_calendar_pronoun)


def _build_step_query(action: str, user_text: str, expected_outcome: str) -> str:
    lowered = re.sub(r"\s+", " ", (user_text or "").strip().lower())
    if action == "google_gmail":
        email_match = re.search(
            r"\bemail\s+(?:to\s+)?([a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,})\b",
            lowered,
            flags=re.IGNORECASE,
        )
        if expected_outcome == "gmail_send" and email_match and re.search(
            r"\b(confirm|confirmation|confirming)\b", lowered
        ):
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
        if expected_outcome == "gmail_send":
            if re.search(r"\b(send|sned|snd|email|compose|write|tell)\b", lowered):
                return user_text.strip()
            return f"send an email based on this request: {user_text.strip()}"
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
        if expected_outcome == "calendar_write":
            cleaned = _build_calendar_write_step_query(user_text)
            if cleaned:
                return cleaned
            return user_text.strip()
        if re.search(r"\b(create|add|reschedule|move|update|book|set up)\b", lowered):
            return f"schedule or update a calendar event based on this request: {user_text.strip()}"
        if "remind" in lowered:
            return f"add a reminder event based on this request: {user_text.strip()}"
        return f"check my calendar events based on this request: {user_text.strip()}"
    return user_text.strip()


def _build_routed_fallback_step(
    *,
    route_action: str,
    text: str,
    why: str = "router_intent_fallback",
) -> OrchestrationStep | None:
    action = (route_action or "").strip().lower()
    if action not in {"google_gmail", "google_calendar", "google_drive", "web_search"}:
        return None

    lowered = re.sub(r"\s+", " ", (text or "").strip().lower())
    expected_outcome = "chat"
    operation = "read"
    if action == "google_gmail":
        is_send = _looks_like_gmail_send_request(lowered)
        expected_outcome = "gmail_send" if is_send else "gmail_read"
        operation = "send" if is_send else "read"
    elif action == "google_calendar":
        is_write = _looks_like_calendar_write_request(lowered)
        expected_outcome = "calendar_write" if is_write else "calendar_read"
        operation = "create" if is_write else "read"
    elif action == "google_drive":
        expected_outcome = "drive_search"
        operation = "search"
    elif action == "web_search":
        expected_outcome = "web_search"
        operation = "search"

    query = _build_step_query(
        action=action,
        user_text=text,
        expected_outcome=expected_outcome,
    )
    requires_confirmation = expected_outcome in {"gmail_send", "calendar_write"}
    return OrchestrationStep(
        step_id="fallback_step_1",
        action=action,
        operation=operation,
        args={},
        query=query,
        expected_outcome=expected_outcome,
        requires_confirmation=requires_confirmation,
        depends_on=[],
        why=why,
    )


def _build_calendar_write_step_query(user_text: str) -> str:
    fields = _extract_calendar_fields(user_text)
    parts: list[str] = ["add event"]
    title = str(fields.get("title") or "").strip()
    day = str(fields.get("day") or "").strip()
    time_value = str(fields.get("time") or "").strip()
    location = str(fields.get("location") or "").strip()
    if title:
        parts.append(title)
    if day:
        parts.append(f"on {day}")
    if time_value:
        parts.append(f"at {time_value}")
    if location:
        parts.append(f"in {location}")
    if len(parts) == 1:
        return ""
    return " ".join(parts)


def _pending_action_from_step_result(
    *, thread_id: str, step: OrchestrationStep, result: OrchestrationStepResult
) -> PendingAction | None:
    reason = (result.reason or "").strip().lower()
    if result.action == "google_calendar" and reason in {
        "calendar_write_confirmation_required",
        "calendar_confirmation_required",
    }:
        pending_args = dict(step.args or {})
        pending_args["draft_text"] = result.assistant_text
        if "event_text" not in pending_args and step.query:
            pending_args["event_text"] = step.query
        return PendingAction(
            pending_action_id=f"pa_{uuid4().hex[:12]}",
            thread_id=thread_id,
            action="google_calendar",
            operation=(step.operation or "create").strip().lower() or "create",
            args=pending_args,
            query=step.query,
            expected_outcome="calendar_write",
            created_at=datetime.now(timezone.utc).isoformat(),
            status="pending",
        )

    if result.action == "google_gmail" and reason in {
        "gmail_send_confirmation_required",
        "gmail_draft_created_not_sent",
    }:
        pending_args = dict(step.args or {})
        draft_id = (
            str(pending_args.get("draft_id") or "").strip()
            or _extract_draft_id_from_text(result.assistant_text)
            or _extract_draft_id_from_text(step.query)
        )
        if draft_id:
            pending_args["draft_id"] = draft_id
        pending_query = f"confirm send draft {draft_id}" if draft_id else step.query
        return PendingAction(
            pending_action_id=f"pa_{uuid4().hex[:12]}",
            thread_id=thread_id,
            action="google_gmail",
            operation="send",
            args=pending_args,
            query=pending_query,
            expected_outcome="gmail_send",
            created_at=datetime.now(timezone.utc).isoformat(),
            status="pending",
        )
    return None


def _build_confirmed_step_from_pending(
    *, action: PendingAction, followup_text: str
) -> OrchestrationStep:
    operation = (action.operation or "").strip().lower()
    expected_outcome = (action.expected_outcome or "").strip().lower()
    query = action.query
    args = {**action.args, "pending_action_id": action.pending_action_id}

    if action.action == "google_gmail":
        draft_id = (
            str(args.get("draft_id") or "").strip()
            or _extract_draft_id_from_text(query)
            or _extract_draft_id_from_text(followup_text)
        )
        if draft_id:
            args["draft_id"] = draft_id
            query = f"confirm send draft {draft_id}"
            operation = "send"
            expected_outcome = "gmail_send"

    if action.action == "google_calendar":
        draft_text = str(args.get("draft_text") or "").strip()
        if draft_text:
            query = _build_confirmed_calendar_request(
                draft_text=draft_text,
                followup_text=followup_text,
            )
            operation = "create"
            expected_outcome = "calendar_write"
        elif expected_outcome == "calendar_write":
            operation = "create"
            if "event_text" not in args:
                args["event_text"] = query

    return OrchestrationStep(
        step_id=action.pending_action_id,
        action=action.action,
        operation=operation,
        args=args,
        query=query,
        expected_outcome=expected_outcome or action.expected_outcome,
        requires_confirmation=False,
        depends_on=[],
        why="confirmed_pending_action",
    )


def _apply_pending_action_edits(
    *, pending_actions: list[PendingAction], selected_ids: set[str], followup_text: str
) -> list[PendingAction]:
    updated: list[PendingAction] = []
    for action in pending_actions:
        if action.pending_action_id not in selected_ids or action.action != "google_calendar":
            updated.append(action)
            continue
        updated.append(
            _apply_single_pending_calendar_edit(action=action, followup_text=followup_text)
        )
    return updated


def _apply_single_pending_calendar_edit(
    *, action: PendingAction, followup_text: str
) -> PendingAction:
    args = dict(action.args)
    draft_text = str(args.get("draft_text") or "").strip()
    if draft_text:
        updated_draft = _apply_calendar_draft_edits(
            draft_text=draft_text,
            followup_text=followup_text,
        )
        args["draft_text"] = updated_draft
        args["event_text"] = _build_confirmed_calendar_request(
            draft_text=updated_draft,
            followup_text="confirm",
        )
        return PendingAction(
            pending_action_id=action.pending_action_id,
            thread_id=action.thread_id,
            action=action.action,
            operation=action.operation,
            args=args,
            query=args["event_text"],
            expected_outcome="calendar_write",
            created_at=action.created_at,
            status=action.status,
        )

    merged_query = _merge_calendar_followup_query(
        base_query=action.query,
        followup_text=followup_text,
    )
    args["event_text"] = merged_query
    return PendingAction(
        pending_action_id=action.pending_action_id,
        thread_id=action.thread_id,
        action=action.action,
        operation=action.operation,
        args=args,
        query=merged_query,
        expected_outcome=action.expected_outcome,
        created_at=action.created_at,
        status=action.status,
    )


def _extract_draft_id_from_text(text: str) -> str | None:
    lowered = (text or "").strip().lower()
    if not lowered:
        return None
    compose_match = re.search(r"compose=([a-z0-9_-]{3,})", lowered)
    if compose_match:
        return compose_match.group(1).strip()
    draft_match = re.search(r"\bdraft(?:\s+id)?\s*[:#]?\s*([a-z0-9_-]{3,})\b", lowered)
    if draft_match:
        return draft_match.group(1).strip()
    return None


def _is_gmail_sent_items(items: list) -> bool:
    for item in items:
        title = str(getattr(item, "title", "")).strip().lower()
        if title.startswith("[sent]"):
            return True
    return False


def _is_gmail_send_confirmation_required_items(items: list) -> bool:
    if len(items) != 1:
        return False
    title = str(getattr(items[0], "title", "")).strip().lower()
    return title in {"send confirmation required", "confirmation required"}


def _is_gmail_drafted_items(items: list) -> bool:
    for item in items:
        title = str(getattr(item, "title", "")).strip().lower()
        if title.startswith("[drafted]"):
            return True
    return False


def _evaluate_calendar_step_execution(
    expected_outcome: str, items: list
) -> tuple[str, str]:
    if expected_outcome == "calendar_write":
        if _is_calendar_created_items(items):
            return ("completed", "calendar_write_completed")
        if _is_calendar_confirmation_required_items(items):
            return ("action_required", "calendar_write_confirmation_required")
        return ("failed", "calendar_write_not_executed")
    if _is_calendar_confirmation_required_items(items):
        return ("action_required", "calendar_confirmation_required")
    return ("completed", "ok")


def _evaluate_gmail_step_execution(
    expected_outcome: str, items: list
) -> tuple[str, str]:
    if expected_outcome == "gmail_send":
        if _is_gmail_sent_items(items):
            return ("completed", "gmail_send_completed")
        if _is_gmail_send_confirmation_required_items(items):
            return ("action_required", "gmail_send_confirmation_required")
        if _is_gmail_drafted_items(items):
            return ("action_required", "gmail_draft_created_not_sent")
        return ("failed", "gmail_send_not_executed")
    if _is_gmail_send_confirmation_required_items(items):
        return ("action_required", "gmail_send_confirmation_required")
    return ("completed", "ok")


def _annotate_orchestration_step_text(
    action: str,
    expected_outcome: str,
    execution_status: str,
    assistant_text: str,
) -> str:
    base_text = (assistant_text or "").strip()
    prefix = ""
    if execution_status == "action_required":
        if action == "google_gmail" and expected_outcome == "gmail_send":
            prefix = (
                "Action required: Gmail needs confirmation before sending. "
                "No email has been sent yet."
            )
        elif action == "google_calendar" and expected_outcome == "calendar_write":
            prefix = (
                "Action required: Google Calendar needs confirmation before creating this event. "
                "No event has been added yet."
            )
        else:
            prefix = "Action required before this step can be fully completed."
    elif execution_status == "failed":
        if action == "google_gmail" and expected_outcome == "gmail_send":
            prefix = "This step did not send the requested email."
        elif action == "google_calendar" and expected_outcome == "calendar_write":
            prefix = "This step did not create the requested calendar event."

    if not prefix:
        return base_text
    if not base_text:
        return prefix
    return f"{prefix}\n\n{base_text}"


def _first_keyword_position(text: str, patterns: list[str]) -> int:
    found_positions: list[int] = []
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            found_positions.append(match.start())
    if not found_positions:
        return -1
    return min(found_positions)


def _step_execution_status(step: OrchestrationStepResult) -> str:
    raw = (step.execution_status or "").strip().lower()
    if raw in {"completed", "action_required", "failed"}:
        return raw
    return "completed" if step.success else "failed"


def _orchestration_status_label(status: str) -> str:
    if status == "completed":
        return "Completed"
    if status == "action_required":
        return "Action Required"
    return "Failed"


def _format_orchestration_response(step_results: list[OrchestrationStepResult]) -> str:
    if not step_results:
        return "No orchestration steps were executed."
    completed = sum(1 for step in step_results if _step_execution_status(step) == "completed")
    action_required = sum(
        1 for step in step_results if _step_execution_status(step) == "action_required"
    )
    failed = sum(1 for step in step_results if _step_execution_status(step) == "failed")
    total = len(step_results)
    lines = [
        "Orchestration Summary",
        "---------------------",
        (
            f"Executed {total} step(s). Completed: {completed}. "
            f"Action required: {action_required}. Failed: {failed}."
        ),
    ]
    if action_required > 0:
        pending_lines = [
            _pending_task_label(step)
            for step in step_results
            if _step_execution_status(step) == "action_required"
        ]
        lines.extend(
            [
                "Pending tasks:",
                *[f"- {pending}" for pending in pending_lines],
                "Reply with 'confirm' to execute all pending tasks.",
                "Reply with 'cancel' to stop all pending tasks.",
                "Or tell me what to change.",
                "",
            ]
        )
    else:
        lines.append("")
    lines.extend(
        [
            "Step Results",
            "------------",
        ]
    )
    for index, step in enumerate(step_results, start=1):
        status = _orchestration_status_label(_step_execution_status(step))
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


def _pending_task_label(step: OrchestrationStepResult) -> str:
    reason = (step.reason or "").strip().lower()
    if step.action == "google_gmail":
        if reason in {"gmail_send_confirmation_required", "gmail_draft_created_not_sent"}:
            return "Email pending confirmation (Gmail)"
        return "Gmail action pending confirmation"
    if step.action == "google_calendar":
        if reason in {"calendar_write_confirmation_required", "calendar_confirmation_required"}:
            return "Calendar event pending confirmation (Google Calendar)"
        return "Google Calendar action pending confirmation"
    if step.action == "google_drive":
        return "Google Drive action pending confirmation"
    return f"{step.capability_label} action pending confirmation"


def _serialize_orchestration_step_result(step: OrchestrationStepResult) -> dict[str, object]:
    return {
        "action": step.action,
        "tool_name": step.tool_name,
        "query": step.query,
        "success": step.success,
        "reason": step.reason,
        "execution_status": _step_execution_status(step),
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
            add_title_match = re.search(
                r"\b(?:add|create|schedule|book|put)\s+(?:an?\s+)?(?:event\s+)?(.+?)(?=\s+(?:to\s+(?:my\s+)?calendar|to\s+(?:my|the)\s+schedule|on|at|for|in)\b|[,.!?]|$)",
                lowered,
            )
            if add_title_match:
                candidate = add_title_match.group(1).strip()
                if candidate.startswith("my meeting with "):
                    out["title"] = "Meeting with " + _title_case_words(
                        candidate[len("my meeting with ") :]
                    )
                else:
                    out["title"] = _title_case_words(candidate)
            else:
                with_match = re.search(
                    r"\bwith\s+([a-z][a-z\s'-]{1,50}?)(?=\s+(?:on|at|in|and|for|its|it'?s|please|add|create|schedule|book)\b|\s+to\s+(?:my|the)\s+(?:calendar|schedule)\b|[,.!?]|$)",
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
        time_match = re.search(r"\b(\d{1,4}(?::\d{2})?\s*(?:am|pm))\b", lowered)
        if time_match:
            out["time"] = time_match.group(1).replace(" ", "")

    if "location" not in out:
        location_match = re.search(
            r"\bin\s+([a-z][a-z\s,.'-]{1,60}?)(?=(?:\s+(?:on|at|add|create|schedule|book|put|to)\b)|$)",
            lowered,
        )
        if location_match:
            out["location"] = _title_case_words(location_match.group(1))
        else:
            at_location_match = re.search(
                r"\bat\s+([a-z][a-z\s,.'-]{1,60}?)(?=(?:\s+(?:on|in|add|create|schedule|book|put|to)\b)|$)",
                lowered,
            )
            if at_location_match:
                out["location"] = _title_case_words(at_location_match.group(1))
    return out


def _build_confirmed_calendar_request(draft_text: str, followup_text: str) -> str:
    base_fields = _extract_calendar_fields(draft_text)
    merged_fields = _merge_calendar_fields(
        base_fields=base_fields,
        followup_text=followup_text,
    )
    return _compose_calendar_confirm_query(merged_fields)


def _apply_calendar_draft_edits(draft_text: str, followup_text: str) -> str:
    base_fields = _extract_calendar_fields(draft_text)
    merged_fields = _merge_calendar_fields(
        base_fields=base_fields,
        followup_text=followup_text,
    )
    return _render_calendar_draft_prompt(merged_fields)


def _merge_calendar_followup_query(base_query: str, followup_text: str) -> str:
    base_fields = _extract_calendar_fields(base_query)
    merged_fields = _merge_calendar_fields(
        base_fields=base_fields,
        followup_text=followup_text,
    )
    return _compose_calendar_action_query(merged_fields)


def _merge_calendar_fields(base_fields: dict[str, str], followup_text: str) -> dict[str, str]:
    merged = {
        "title": str(base_fields.get("title") or "").strip(),
        "day": str(base_fields.get("day") or "").strip(),
        "time": str(base_fields.get("time") or "").strip(),
        "location": str(base_fields.get("location") or "").strip(),
    }
    followup_fields = _extract_calendar_fields(followup_text)
    for key in ("title", "day", "time", "location"):
        candidate = str(followup_fields.get(key) or "").strip()
        if candidate:
            merged[key] = candidate
    overrides = _extract_calendar_followup_overrides(followup_text)
    for key, value in overrides.items():
        if value:
            merged[key] = value
    return merged


def _extract_calendar_followup_overrides(followup_text: str) -> dict[str, str]:
    lowered = re.sub(r"\s+", " ", (followup_text or "").strip().lower())
    out: dict[str, str] = {}
    if not lowered:
        return out

    title_match = re.search(
        r"\b(?:title|name)\s+(?:to|as|is|should\s+be|should\s+just\s+be)\s+(.+?)(?=$|[,.!?])",
        lowered,
    )
    if title_match:
        out["title"] = _title_case_words(title_match.group(1))

    location_match = re.search(
        r"\b(?:location|place|venue)\s+(?:to|as|is|should\s+be|should\s+just\s+be)\s+(.+?)(?=$|[,.!?])",
        lowered,
    )
    if location_match:
        out["location"] = _title_case_words(location_match.group(1))

    return out


def _compose_calendar_confirm_query(fields: dict[str, str]) -> str:
    parts = ["confirm: add event"]
    title = str(fields.get("title") or "").strip()
    day = str(fields.get("day") or "").strip()
    time_value = str(fields.get("time") or "").strip().replace(" ", "")
    location = str(fields.get("location") or "").strip()
    if title:
        parts.append(title)
    if day:
        parts.append(f"on {day}")
    if time_value:
        parts.append(f"at {time_value}")
    if location:
        parts.append(f"in {location}")
    return " ".join(parts).strip()


def _compose_calendar_action_query(fields: dict[str, str]) -> str:
    parts = ["add event"]
    title = str(fields.get("title") or "").strip()
    day = str(fields.get("day") or "").strip()
    time_value = str(fields.get("time") or "").strip().replace(" ", "")
    location = str(fields.get("location") or "").strip()
    if title:
        parts.append(title)
    if day:
        parts.append(f"on {day}")
    if time_value:
        parts.append(f"at {time_value}")
    if location:
        parts.append(f"in {location}")
    return " ".join(parts).strip()


def _render_calendar_draft_prompt(fields: dict[str, str]) -> str:
    title = str(fields.get("title") or "Meeting").strip()
    day = str(fields.get("day") or "unspecified").strip()
    time_value = str(fields.get("time") or "unspecified").strip().replace(" ", "")
    location = str(fields.get("location") or "").strip()
    time_display = time_value
    if time_display and (time_display.endswith("am") or time_display.endswith("pm")):
        time_display = time_display[:-2] + " " + time_display[-2:].upper()
    lines = [
        "I have this draft event:",
        f"- Title: {title}",
        f"- Day: {day}",
        f"- Time: {time_display or 'unspecified'}",
    ]
    if location:
        lines.append(f"- Location: {location}")
    lines.append(
        "Should I add this to Google Calendar? Reply with 'confirm' to proceed or 'cancel' to stop."
    )
    return "\n".join(lines)


def _title_case_words(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value.strip(" ."))
    return " ".join(part.capitalize() for part in cleaned.split(" ") if part)


def _is_write_step(step: OrchestrationStep) -> bool:
    outcome = (step.expected_outcome or "").strip().lower()
    operation = (step.operation or "").strip().lower()
    if outcome in {"gmail_send", "calendar_write"}:
        return True
    return operation in {"send", "create", "write"}


def _make_pending_action(thread_id: str, step: OrchestrationStep) -> PendingAction:
    pending_action_id = f"pa_{uuid4().hex[:12]}"
    derived_args = _derive_step_args(step)
    return PendingAction(
        pending_action_id=pending_action_id,
        thread_id=thread_id,
        action=step.action,
        operation=(step.operation or "").strip().lower(),
        args={**derived_args, **dict(step.args or {})},
        query=step.query,
        expected_outcome=step.expected_outcome,
        created_at=datetime.now(timezone.utc).isoformat(),
        status="pending",
    )


def _extract_pending_action_id(text: str) -> str | None:
    match = re.search(r"\bpa_[a-z0-9]{6,32}\b", (text or "").strip().lower())
    if not match:
        return None
    return match.group(0)


def _select_pending_actions(
    *, pending_actions: list[PendingAction], pending_action_id: str | None
) -> list[PendingAction]:
    if not pending_action_id:
        return pending_actions
    selected = [
        action
        for action in pending_actions
        if action.pending_action_id == pending_action_id
    ]
    return selected or pending_actions


def _render_pending_action_created_message(pending: PendingAction) -> str:
    label = _capability_label_for_action(pending.action)
    if pending.action == "google_calendar" and pending.expected_outcome == "calendar_write":
        draft_prompt = _render_calendar_draft_prompt(_extract_calendar_fields(pending.query))
        return (
            f"{draft_prompt}\n\n"
            f"pending_action_id={pending.pending_action_id}. "
            "Reply with 'confirm' to execute or 'cancel' to drop it."
        )
    if pending.action == "google_gmail" and pending.expected_outcome == "gmail_send":
        draft_id = (
            str(pending.args.get("draft_id") or "").strip()
            or _extract_draft_id_from_text(pending.query)
        )
        target = f" draft {draft_id}" if draft_id else " draft"
        return (
            f"I have a pending Gmail send for{target}.\n"
            f"pending_action_id={pending.pending_action_id}. "
            "Reply with 'confirm' to execute or 'cancel' to drop it."
        )
    return (
        f"Action requires confirmation ({label}). "
        f"pending_action_id={pending.pending_action_id}. "
        "Reply with 'confirm' to execute or 'cancel' to drop it."
    )


def _render_pending_actions_message(pending_actions: list[PendingAction]) -> str:
    lines = [
        "Pending actions waiting for confirmation:",
    ]
    for action in pending_actions:
        lines.append(
            f"- {action.pending_action_id}: {_capability_label_for_action(action.action)} ({action.operation or 'operation_unspecified'})"
        )
    lines.append("Reply with 'confirm' to execute all, or include a pending_action_id to target one action.")
    return "\n".join(lines)


def _render_pending_actions_status(*, pending_actions: list[PendingAction]) -> str:
    count = len(pending_actions)
    noun = "action" if count == 1 else "actions"
    lines = [f"You currently have {count} pending {noun} waiting for confirmation."]
    lines.append("")
    lines.append(_render_pending_actions_message(pending_actions))
    return "\n".join(lines).strip()


def _looks_like_pending_status_query(text: str) -> bool:
    lowered = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not lowered:
        return False
    if re.search(r"\b(how many|count|total|pending|drafts?|only one|just one|was that)\b", lowered):
        return True
    return False


def _looks_like_new_request_while_pending(*, text: str, pending_intent: str) -> bool:
    if pending_intent != "unknown":
        return False
    lowered = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not lowered:
        return False
    asks_calendar_read = bool(
        re.search(r"\b(check|show|list|what|when|view|look|see)\b", lowered)
        and re.search(r"\b(calendar|schedule|events?|meetings?)\b", lowered)
    )
    asks_time_range = bool(
        re.search(r"\b(today|tomorrow|week|month|this month|next month|upcoming)\b", lowered)
        and re.search(r"\b(calendar|schedule|events?|meetings?)\b", lowered)
    )
    return asks_calendar_read or asks_time_range


def _render_no_pending_status(*, text: str, last_cleared_count: int | None) -> str:
    lowered = re.sub(r"\s+", " ", (text or "").strip().lower())
    asks_single = bool(re.search(r"\b(only one|just one|was that the only one|so just one)\b", lowered))
    if asks_single and last_cleared_count is not None:
        if last_cleared_count == 1:
            return "Yes, that was the only pending action, and it is now cleared."
        return (
            f"No. There were {last_cleared_count} pending actions in that set, "
            "and they are now cleared."
        )
    if last_cleared_count is not None:
        noun = "action" if last_cleared_count == 1 else "actions"
        return (
            "There are no pending actions waiting for confirmation right now. "
            f"The last cleared set had {last_cleared_count} pending {noun}."
        )
    return "There are no pending actions waiting for confirmation right now."


def _derive_step_args(step: OrchestrationStep) -> dict[str, object]:
    action = (step.action or "").strip().lower()
    query = (step.query or "").strip()
    lowered = query.lower()
    if action == "google_gmail":
        draft_match = re.search(r"\bdraft(?:\s+id)?\s*[:#]?\s*([a-z0-9_-]{3,})\b", lowered)
        if draft_match:
            return {"draft_id": draft_match.group(1).strip()}
    if action == "google_calendar":
        return {"event_text": query}
    return {}


def _prevent_unexecuted_action_claims(
    assistant_text: str, routed_action: str, user_text: str = ""
) -> str:
    if routed_action in {"google_calendar", "google_gmail", "google_drive"}:
        return assistant_text
    lowered = (assistant_text or "").strip().lower()
    if not lowered:
        return assistant_text
    lowered_user = (user_text or "").strip().lower()
    strict_calendar_read_request = bool(
        re.search(r"\b(calendar|schedule|agenda|events?|meetings?)\b", lowered_user)
        and re.search(
            r"\b(check|show|list|what|when|month|week|today|tomorrow|upcoming|agenda)\b",
            lowered_user,
        )
    )
    asks_calendar_read = bool(
        re.search(r"\b(calendar|schedule|events?|meetings?)\b", lowered_user)
        and re.search(r"\b(check|show|list|what|when|month|week|today|tomorrow|upcoming)\b", lowered_user)
    )
    likely_fabricated_calendar_read = bool(
        re.search(
            r"\byou have (?:the )?following (?:events|deadlines|tasks)\b",
            lowered,
        )
        or re.search(r"\brecurring (?:event|tasks?)\b", lowered)
        or re.search(r"^\s*\d+\.\s*(meeting|task|deadline|event)\b", lowered, re.MULTILINE)
    )
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
    if strict_calendar_read_request:
        return (
            "I can’t provide live calendar results from memory-only mode. "
            "I need to run Google Calendar to list your real schedule."
        )
    if asks_calendar_read and likely_fabricated_calendar_read:
        return (
            "I can’t verify live calendar events from memory-only mode. "
            "I need to run Google Calendar to list your real schedule."
        )
    if gmail_claim:
        return (
            "I haven’t sent an email yet. "
            "I can draft it and send only after your confirmation."
        )
    return assistant_text
