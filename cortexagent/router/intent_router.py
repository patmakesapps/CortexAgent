from dataclasses import dataclass
import json
from urllib import error as urlerror
from urllib import request as urlrequest

from cortexagent.config import settings

WEB_CUE_TERMS = (
    "latest",
    "today",
    "news",
    "current",
    "price",
    "look up",
    "search",
    "web",
    "online",
    "recent",
    "release",
    "update",
)


@dataclass(frozen=True)
class RouteDecision:
    action: str
    reason: str
    confidence: float


def _keyword_decision(user_text: str) -> RouteDecision:
    text = user_text.strip().lower()
    if any(term in text for term in WEB_CUE_TERMS):
        return RouteDecision(
            action="web_search",
            reason="matched_web_cue",
            confidence=0.9,
        )
    return RouteDecision(action="chat", reason="default_chat", confidence=0.75)


def _extract_json(content: str) -> dict[str, object]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1]).strip()
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("Router JSON payload was not an object.")
    return value


def _llm_decision(user_text: str) -> RouteDecision | None:
    if not settings.router_llm_enabled:
        return None
    if not settings.groq_api_key:
        return None

    prompt = (
        "You are a routing classifier for a chat agent.\n"
        "Choose one action for the user request:\n"
        "- web_search: requires current/external/verifiable web data\n"
        "- chat: answer directly from internal reasoning/knowledge\n\n"
        "Rules:\n"
        "- Prefer web_search for time-sensitive facts (news, prices, weather, sports, schedules, releases, laws, policies, outages, version changes).\n"
        "- Prefer web_search when stale information could cause a wrong answer.\n"
        "- Prefer chat for timeless concepts, creative writing, coding help, editing, translation, and personal advice.\n"
        "- Keywords are hints, not the source of truth.\n"
        "- Return strict JSON only.\n\n"
        "Output schema:\n"
        '{"action":"web_search|chat","reason":"short_reason","confidence":0.0}'
    )

    payload = {
        "model": settings.router_llm_model,
        "temperature": 0,
        "max_tokens": 120,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_text.strip()},
        ],
    }
    body = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.groq_api_key}",
        },
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=settings.router_llm_timeout_seconds) as res:
            response_payload = json.loads(res.read().decode("utf-8"))
    except (urlerror.HTTPError, urlerror.URLError, TimeoutError, json.JSONDecodeError):
        return None
    except Exception:
        return None

    try:
        content = (
            response_payload.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        data = _extract_json(content)
        action = str(data.get("action", "")).strip().lower()
        reason = str(data.get("reason", "")).strip() or "llm_router"
        confidence = float(data.get("confidence", 0.7))
    except Exception:
        return None

    if action not in {"chat", "web_search"}:
        return None
    if confidence < 0.0:
        confidence = 0.0
    if confidence > 1.0:
        confidence = 1.0
    return RouteDecision(action=action, reason=f"llm:{reason}", confidence=confidence)


def decide_action(user_text: str, tools_enabled: bool, web_search_enabled: bool) -> RouteDecision:
    text = user_text.strip().lower()
    if not text:
        return RouteDecision(action="chat", reason="empty_text", confidence=1.0)

    if not tools_enabled:
        return RouteDecision(action="chat", reason="tools_disabled", confidence=1.0)

    if web_search_enabled:
        llm_route = _llm_decision(user_text)
        if llm_route is not None:
            return llm_route
        return _keyword_decision(user_text)

    return RouteDecision(action="chat", reason="default_chat", confidence=0.75)
