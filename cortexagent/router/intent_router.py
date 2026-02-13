from dataclasses import dataclass


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


def decide_action(user_text: str, tools_enabled: bool, web_search_enabled: bool) -> RouteDecision:
    text = user_text.strip().lower()
    if not text:
        return RouteDecision(action="chat", reason="empty_text", confidence=1.0)

    if not tools_enabled:
        return RouteDecision(action="chat", reason="tools_disabled", confidence=1.0)

    if web_search_enabled and any(term in text for term in WEB_CUE_TERMS):
        return RouteDecision(
            action="web_search",
            reason="matched_web_cue",
            confidence=0.9,
        )

    return RouteDecision(action="chat", reason="default_chat", confidence=0.75)
