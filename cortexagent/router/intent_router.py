from dataclasses import dataclass

from cortexagent.config import settings
from cortexagent.services.llm_json_client import call_json_chat_completion


@dataclass(frozen=True)
class RouteDecision:
    action: str
    reason: str
    confidence: float


def _llm_decision(
    user_text: str,
    prior_user_text: str | None = None,
    prior_tool_action: str | None = None,
) -> RouteDecision | None:
    if not settings.router_llm_enabled:
        return None
    response = call_json_chat_completion(
        provider=settings.router_llm_provider,
        model=settings.router_llm_model,
        api_key=settings.router_llm_api_key,
        timeout_seconds=settings.router_llm_timeout_seconds,
        api_base_url=settings.router_llm_api_base_url,
        system_prompt=(
            "You are a routing classifier.\n"
            "Return strict JSON with keys: action, reason, confidence, needs_external_data.\n"
            "Allowed actions: web_search, google_calendar, google_drive, google_gmail, chat."
        ),
        user_prompt=(
            f"Current user message:\n{(user_text or '').strip()}\n\n"
            f"Prior user message:\n{(prior_user_text or '').strip() or 'none'}\n\n"
            f"Prior routed tool action:\n{(prior_tool_action or '').strip() or 'none'}"
        ),
        max_tokens=120,
        temperature=0.0,
    )
    if response.error is not None or not response.data:
        return None
    action = str(response.data.get("action") or "").strip().lower()
    if action not in {"chat", "web_search", "google_calendar", "google_drive", "google_gmail"}:
        return None
    reason = str(response.data.get("reason") or "").strip() or "llm_router"
    confidence = _coerce_confidence(response.data.get("confidence"))
    needs_external_data = bool(response.data.get("needs_external_data", False))
    if needs_external_data and action == "chat":
        action = "web_search"
        reason = f"{reason}|external_data_required"
        confidence = max(confidence, 0.8)
    return RouteDecision(action=action, reason=f"llm:{reason}", confidence=confidence)


def _coerce_confidence(raw: object) -> float:
    try:
        value = float(raw)
    except Exception:
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _apply_route_policy(route: RouteDecision, web_search_enabled: bool) -> RouteDecision:
    if not web_search_enabled and route.action == "web_search":
        return RouteDecision(action="chat", reason="web_search_disabled", confidence=0.9)
    return route


def decide_action(
    user_text: str,
    tools_enabled: bool,
    web_search_enabled: bool,
    prior_user_text: str | None = None,
    prior_tool_action: str | None = None,
) -> RouteDecision:
    text = user_text.strip().lower()
    if not text:
        return RouteDecision(action="chat", reason="empty_text", confidence=1.0)
    if not tools_enabled:
        return RouteDecision(action="chat", reason="tools_disabled", confidence=1.0)
    llm_route = _llm_decision(
        user_text=user_text,
        prior_user_text=prior_user_text,
        prior_tool_action=prior_tool_action,
    )
    if llm_route is None:
        return RouteDecision(action="chat", reason="llm_router_unavailable", confidence=0.0)
    return _apply_route_policy(llm_route, web_search_enabled=web_search_enabled)
