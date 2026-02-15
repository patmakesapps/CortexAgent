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
    "link me",
    "link me to",
    "can you link me",
    "give me links",
    "send me links",
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
    "hotel",
    "flight",
    "airfare",
    "fare",
    "rate",
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

GOOGLE_DRIVE_TERMS = (
    "google drive",
    "drive",
    "my files",
    "shared with me",
    "find file",
    "find files",
    "search drive",
    "open drive",
    "drive docs",
    "drive folders",
)


@dataclass(frozen=True)
class RouteDecision:
    action: str
    reason: str
    confidence: float


def _keyword_decision(user_text: str) -> RouteDecision:
    text = user_text.strip().lower()
    if _looks_like_link_request_followup(text):
        return RouteDecision(
            action="web_search",
            reason="matched_link_request_followup",
            confidence=0.95,
        )
    if _looks_like_shopping_research_request(text):
        return RouteDecision(
            action="web_search",
            reason="matched_shopping_research_request",
            confidence=0.94,
        )
    if _looks_like_travel_price_request(text):
        return RouteDecision(
            action="web_search",
            reason="matched_travel_price_request",
            confidence=0.95,
        )
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
    if _matches_explicit_drive_intent(text):
        return RouteDecision(
            action="google_drive",
            reason="matched_google_drive_intent",
            confidence=0.94,
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
    if any(term in text for term in GOOGLE_DRIVE_TERMS):
        return RouteDecision(
            action="google_drive",
            reason="matched_google_drive_term",
            confidence=0.91,
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


def _looks_like_travel_price_request(text: str) -> bool:
    if not text:
        return False
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    has_price_signal = bool(
        re.search(r"\b(price|prices|rate|rates|cost|costs|cheap|cheaper|best deal|deals)\b", normalized)
    )
    has_travel_entity = bool(
        re.search(r"\b(hotel|hotels|flight|flights|airfare|air fare|lodging|stay)\b", normalized)
    )
    has_location_hint = bool(
        re.search(r"\b(in|near|around|for)\s+[a-z]{2,}", normalized)
        or " dc " in f" {normalized} "
        or "washington dc" in normalized
    )
    return has_price_signal and has_travel_entity and has_location_hint


def _looks_like_shopping_research_request(text: str) -> bool:
    if not text:
        return False
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    has_product_intent = bool(
        re.search(
            r"\b(buy|purchase|get|recommend|best|good|options|models|camera|laptop|phone|headphones|monitor|mic|microphone)\b",
            normalized,
        )
    )
    has_price_or_budget = bool(
        re.search(r"\b(price|prices|budget|\$\s*\d+|\d{3,5}\s*(usd|dollars)?)\b", normalized)
    )
    asks_for_external_choices = bool(
        re.search(r"\b(link|links|compare|reviews?|in my price range|for my budget|under \$?\d+)\b", normalized)
    )
    return has_product_intent and (has_price_or_budget or asks_for_external_choices)


def _looks_like_link_request_followup(text: str) -> bool:
    if not text:
        return False
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return bool(
        re.search(
            r"\b(can you|could you|please)?\s*(link me|send me links|give me links|show links)\b",
            normalized,
        )
        or re.search(r"\b(link(s)? to (a few|some|options|models))\b", normalized)
    )


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
        r"\b(add|ad|create|schedule|book|set up|put|reschedule|move|shift|update|change)\b.*\b(my|the|this|that|it)\b.*\bcalendar\b",
        text,
    ):
        return True
    if re.search(
        r"\b(reschedule|move|shift|update|change)\b.*\b(meeting|event|appointment)\b",
        text,
    ):
        return True
    if re.search(
        r"\b(reschedule|move|shift|update|change)\b.*\b(that|this|it)\b.*\b(to|for)\b.*\b\d{1,2}(?::\d{2})?\s*(am|pm)\b",
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


def _matches_explicit_drive_intent(text: str) -> bool:
    if not text:
        return False
    return bool(
        re.search(r"\b(find|search|look for|open|show|list|check)\b.*\b(file|files|folder|folders|doc|document|sheet|slides)\b", text)
        and re.search(r"\b(drive|google drive)\b", text)
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


def _llm_decision(
    user_text: str,
    prior_user_text: str | None = None,
    prior_tool_action: str | None = None,
) -> RouteDecision | None:
    if not settings.router_llm_enabled:
        return None
    if not settings.groq_api_key:
        return None

    prompt = (
        "You are a routing classifier for a chat agent.\n"
        "Available tools:\n"
        "- web_search: use for current external facts, product links/options, prices, comparisons, news\n"
        "- google_calendar: use for user's schedule/events and calendar write actions\n"
        "- google_drive: use for user's files/docs/folders\n"
        "- google_gmail: use for user's inbox/threads/messages/drafts/send flows\n"
        "- chat: use when no external or personal-tool data is needed\n\n"
        "Task:\n"
        "Choose exactly one action for this turn based on user intent and recent context.\n\n"
        "Rules:\n"
        "- Prefer web_search for requests requiring links, options, prices, current facts, or comparisons.\n"
        "- Prefer web_search for follow-ups like 'show me more options', 'link me a few', when prior topic needs external data.\n"
        "- Prefer chat for timeless conceptual guidance when no retrieval is required.\n"
        "- Prefer google_* tools only when user intent is clearly about personal connected data.\n"
        "- Return strict JSON only.\n\n"
        "Output schema:\n"
        '{"action":"web_search|google_calendar|google_drive|google_gmail|chat","reason":"short_reason","confidence":0.0,"needs_external_data":true}'
    )

    payload = {
        "model": settings.router_llm_model,
        "temperature": 0,
        "max_tokens": 120,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Current user message:\n{user_text.strip()}\n\n"
                    f"Prior user message (if any):\n{(prior_user_text or '').strip() or 'none'}\n\n"
                    f"Prior routed tool action (if any):\n{(prior_tool_action or '').strip() or 'none'}"
                ),
            },
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
        needs_external_data = bool(data.get("needs_external_data", False))
    except Exception:
        return None

    if action not in {"chat", "web_search", "google_calendar", "google_drive", "google_gmail"}:
        return None
    if needs_external_data and action == "chat":
        # Keep outputs deterministic: if model says external data is needed, route web_search.
        action = "web_search"
        reason = f"{reason}|external_data_required"
        confidence = max(confidence, 0.8)
    if confidence < 0.0:
        confidence = 0.0
    if confidence > 1.0:
        confidence = 1.0
    return RouteDecision(action=action, reason=f"llm:{reason}", confidence=confidence)


def _apply_route_policy(
    route: RouteDecision,
    user_text: str,
    prior_user_text: str | None,
    prior_tool_action: str | None,
    web_search_enabled: bool,
) -> RouteDecision:
    text = user_text.strip().lower()
    prior_text = (prior_user_text or "").strip().lower()
    if not web_search_enabled:
        return route

    # Deterministic safety net: product-shopping follow-ups should not silently stay in chat.
    if route.action == "chat":
        if _looks_like_shopping_research_request(text):
            return RouteDecision(
                action="web_search",
                reason="policy_override_shopping_research",
                confidence=max(route.confidence, 0.9),
            )
        if _looks_like_link_request_followup(text) and _looks_like_shopping_research_request(prior_text):
            return RouteDecision(
                action="web_search",
                reason="policy_override_link_followup",
                confidence=max(route.confidence, 0.92),
            )
        if prior_tool_action == "web_search" and _looks_like_link_request_followup(text):
            return RouteDecision(
                action="web_search",
                reason="policy_override_continue_web_search",
                confidence=max(route.confidence, 0.9),
            )
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
    if _matches_explicit_drive_intent(text):
        return RouteDecision(
            action="google_drive",
            reason="matched_google_drive_intent",
            confidence=0.94,
        )

    if web_search_enabled:
        llm_route = _llm_decision(
            user_text=user_text,
            prior_user_text=prior_user_text,
            prior_tool_action=prior_tool_action,
        )
        if llm_route is not None:
            return _apply_route_policy(
                llm_route,
                user_text=user_text,
                prior_user_text=prior_user_text,
                prior_tool_action=prior_tool_action,
                web_search_enabled=web_search_enabled,
            )
        keyword_route = _keyword_decision(user_text)
        return _apply_route_policy(
            keyword_route,
            user_text=user_text,
            prior_user_text=prior_user_text,
            prior_tool_action=prior_tool_action,
            web_search_enabled=web_search_enabled,
        )

    return RouteDecision(action="chat", reason="default_chat", confidence=0.75)
