from dataclasses import dataclass
import json
import re
from urllib import error as urlerror
from urllib import request as urlrequest

from cortexagent.config import settings

EXPLICIT_WEB_INTENT_TERMS = (
    "web search",
    "search the web",
    "search online",
    "look up",
    "lookup",
    "google",
    "check online",
    "find online",
    "show links",
    "send links",
    "show sources",
    "source links",
    "citations",
)

TIME_SENSITIVE_FACT_TERMS = (
    "latest",
    "today",
    "right now",
    "current",
    "live",
    "market",
    "market now",
    "news",
    "price",
    "stock",
    "crypto",
    "weather",
    "score",
    "schedule",
    "release date",
    "version",
    "outage",
    "policy change",
)

NON_WEB_TASK_TERMS = (
    "write code",
    "debug",
    "refactor",
    "explain this code",
    "summarize",
    "rewrite",
    "translate",
    "brainstorm",
    "draft",
)

GOOGLE_CALENDAR_TERMS = (
    "calendar",
    "google calendar",
    "upcoming events",
    "my schedule",
    "my events",
    "next meeting",
    "today on my calendar",
    "tomorrow on my calendar",
    "add event",
    "create event",
    "schedule meeting",
    "add to my calendar",
    "put on my calendar",
)

GOOGLE_GMAIL_TERMS = (
    "gmail",
    "email",
    "inbox",
    "recent threads",
    "recent emails",
    "read message",
    "read email",
    "open email",
    "open message",
    "draft reply",
    "draft email",
    "compose reply",
    "send draft",
)


@dataclass(frozen=True)
class RouteDecision:
    action: str
    reason: str
    confidence: float


def _keyword_decision(user_text: str) -> RouteDecision:
    text = user_text.strip().lower()
    if _matches_explicit_calendar_write_intent(text):
        return RouteDecision(
            action="google_calendar",
            reason="matched_explicit_google_calendar_write_intent",
            confidence=0.97,
        )
    if _matches_explicit_gmail_intent(text):
        return RouteDecision(
            action="google_gmail",
            reason="matched_google_gmail_intent",
            confidence=0.95,
        )
    if any(term in text for term in GOOGLE_CALENDAR_TERMS):
        return RouteDecision(
            action="google_calendar",
            reason="matched_google_calendar_intent",
            confidence=0.93,
        )
    if any(term in text for term in GOOGLE_GMAIL_TERMS):
        return RouteDecision(
            action="google_gmail",
            reason="matched_google_gmail_term",
            confidence=0.92,
        )
    if any(term in text for term in EXPLICIT_WEB_INTENT_TERMS):
        return RouteDecision(
            action="web_search",
            reason="matched_explicit_web_intent",
            confidence=0.95,
        )
    if any(term in text for term in NON_WEB_TASK_TERMS):
        return RouteDecision(action="chat", reason="matched_non_web_task", confidence=0.9)
    if _looks_like_time_sensitive_fact_request(text):
        return RouteDecision(action="web_search", reason="matched_time_sensitive_fact", confidence=0.84)
    return RouteDecision(action="chat", reason="default_chat", confidence=0.75)


def _looks_like_time_sensitive_fact_request(text: str) -> bool:
    if not text:
        return False
    if _looks_like_live_market_prompt(text):
        return True
    question_like = (
        "?" in text
        or text.startswith(("what", "who", "when", "where", "how", "is", "are", "does", "do", "did", "can"))
        or "tell me" in text
    )
    if not question_like:
        return False
    return any(term in text for term in TIME_SENSITIVE_FACT_TERMS)


def _looks_like_live_market_prompt(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    if not normalized:
        return False
    has_market_subject = bool(
        re.search(r"\b(bitcoin|btc|ethereum|eth|crypto|stock|sp500|nasdaq|dow)\b", normalized)
    )
    has_live_cue = bool(
        re.search(r"\b(live|current|right now|market|price|quote|trading at|doing on the market)\b", normalized)
    )
    return has_market_subject and has_live_cue


def _matches_explicit_calendar_write_intent(text: str) -> bool:
    if not text:
        return False
    if re.search(
        r"\b(add|ad|create|schedule|book|set up|put)\b.*\b(my|the|this|that|it)\b.*\bcalendar\b",
        text,
    ):
        return True
    pattern = re.compile(
        r"\b(add|ad|create|schedule|book|set up)\b.*\b(meeting|event|appointment)\b"
    )
    return bool(pattern.search(text))


def _matches_explicit_gmail_intent(text: str) -> bool:
    if not text:
        return False
    return bool(
        re.search(r"\b(read|open|show|list|check)\b.*\b(email|inbox|thread|message)\b", text)
        or re.search(r"\b(draft|compose|write)\b.*\b(reply|email)\b", text)
        or re.search(r"\bsend\b.*\b(email|gmail|message)\b", text)
        or re.search(r"\b(sned|snd)\b.*\b(email|gmail|message)\b", text)
        or re.search(r"\bsend\b.*\bdraft\b", text)
        or re.search(r"\b(send|sned|snd)\b.*[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}", text)
    )


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
        "- google_calendar: user asks about their own schedule/events/calendar\n"
        "- google_gmail: user asks about their own inbox/threads/messages/drafts\n"
        "- chat: answer directly from internal reasoning/knowledge\n\n"
        "Rules:\n"
        "- Prefer web_search for time-sensitive facts (news, prices, weather, sports, schedules, releases, laws, policies, outages, version changes).\n"
        "- Prefer web_search when stale information could cause a wrong answer.\n"
        "- Prefer chat for timeless concepts, creative writing, coding help, editing, translation, and personal advice.\n"
        "- Keywords are hints, not the source of truth.\n"
        "- If user clearly refers to their personal calendar/events/schedule, prefer google_calendar.\n"
        "- If user clearly refers to their inbox/emails/threads/drafts, prefer google_gmail.\n"
        "- Return strict JSON only.\n\n"
        "Output schema:\n"
        '{"action":"web_search|google_calendar|google_gmail|chat","reason":"short_reason","confidence":0.0}'
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

    if action not in {"chat", "web_search", "google_calendar", "google_gmail"}:
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

    if _matches_explicit_calendar_write_intent(text):
        return RouteDecision(
            action="google_calendar",
            reason="matched_explicit_google_calendar_write_intent",
            confidence=0.97,
        )
    if _matches_explicit_gmail_intent(text):
        return RouteDecision(
            action="google_gmail",
            reason="matched_google_gmail_intent",
            confidence=0.95,
        )

    if web_search_enabled:
        llm_route = _llm_decision(user_text)
        if llm_route is not None:
            return llm_route
        return _keyword_decision(user_text)

    return RouteDecision(action="chat", reason="default_chat", confidence=0.75)
