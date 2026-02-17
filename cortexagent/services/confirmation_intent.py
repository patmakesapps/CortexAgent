from dataclasses import dataclass

from cortexagent.config import settings
from cortexagent.services.llm_json_client import call_json_chat_completion

_VALID_INTENTS = {"confirm", "cancel", "pause", "status", "edit", "unknown"}

_DAY_TOKENS = {
    "today",
    "tomorrow",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
}


@dataclass(frozen=True)
class ConfirmationIntent:
    intent: str
    confidence: float
    source: str
    normalized_text: str
    reason: str


def classify_pending_confirmation_intent(
    *,
    text: str,
    pending_calendar: bool,
    pending_gmail: bool,
) -> ConfirmationIntent:
    normalized = _normalize_reply_text(text)
    if not normalized:
        return ConfirmationIntent(
            intent="unknown",
            confidence=0.0,
            source="deterministic",
            normalized_text=normalized,
            reason="empty_reply",
        )
    if not (pending_calendar or pending_gmail):
        return ConfirmationIntent(
            intent="unknown",
            confidence=0.0,
            source="deterministic",
            normalized_text=normalized,
            reason="no_pending_confirmation",
        )

    llm_result = _llm_classify(
        text=text,
        normalized_text=normalized,
        pending_calendar=pending_calendar,
        pending_gmail=pending_gmail,
    )
    if llm_result is not None:
        return llm_result
    return classify_confirmation_intent_deterministic(
        text=text,
        pending_calendar=pending_calendar,
        pending_gmail=pending_gmail,
    )


def classify_confirmation_intent_deterministic(
    *,
    text: str,
    pending_calendar: bool,
    pending_gmail: bool,
) -> ConfirmationIntent:
    normalized = _normalize_reply_text(text)
    if not normalized:
        return ConfirmationIntent(
            intent="unknown",
            confidence=0.0,
            source="deterministic",
            normalized_text=normalized,
            reason="empty_reply",
        )

    tokens = normalized.split(" ")
    token_set = {token for token in tokens if token}

    if token_set.intersection({"cancel", "stop", "no", "nope", "nah"}):
        return ConfirmationIntent(
            intent="cancel",
            confidence=0.9,
            source="deterministic",
            normalized_text=normalized,
            reason="cancel_token",
        )

    if token_set.intersection({"pause", "later", "wait", "hold"}):
        return ConfirmationIntent(
            intent="pause",
            confidence=0.84,
            source="deterministic",
            normalized_text=normalized,
            reason="pause_token",
        )

    if pending_gmail and token_set.intersection({"status", "pending", "sent"}):
        return ConfirmationIntent(
            intent="status",
            confidence=0.82,
            source="deterministic",
            normalized_text=normalized,
            reason="status_token",
        )

    if pending_calendar and (
        token_set.intersection({"edit", "change", "update", "move", "shift", "instead"})
        or token_set.intersection(_DAY_TOKENS)
        or any(_looks_like_time_token(token) for token in tokens)
    ):
        return ConfirmationIntent(
            intent="edit",
            confidence=0.8,
            source="deterministic",
            normalized_text=normalized,
            reason="calendar_edit_token",
        )

    if token_set.intersection({"confirm", "yes", "ok", "okay", "sure", "proceed"}):
        return ConfirmationIntent(
            intent="confirm",
            confidence=0.88,
            source="deterministic",
            normalized_text=normalized,
            reason="confirm_token",
        )

    if (
        len(tokens) <= 3
        and pending_gmail
        and token_set.intersection({"send"})
    ):
        return ConfirmationIntent(
            intent="confirm",
            confidence=0.82,
            source="deterministic",
            normalized_text=normalized,
            reason="short_send_confirm",
        )

    if (
        len(tokens) <= 3
        and pending_calendar
        and token_set.intersection({"add"})
    ):
        return ConfirmationIntent(
            intent="confirm",
            confidence=0.82,
            source="deterministic",
            normalized_text=normalized,
            reason="short_add_confirm",
        )

    return ConfirmationIntent(
        intent="unknown",
        confidence=0.0,
        source="deterministic",
        normalized_text=normalized,
        reason="no_clear_intent",
    )


def _normalize_reply_text(text: str) -> str:
    raw = (text or "").strip().lower()
    if not raw:
        return ""
    raw = raw.replace("’", "'")
    for ch in ".!?":
        raw = raw.replace(ch, " ")
    while "  " in raw:
        raw = raw.replace("  ", " ")
    return raw.strip()


def _looks_like_time_token(token: str) -> bool:
    candidate = (token or "").strip().lower().replace(",", "")
    if not candidate:
        return False
    if candidate.endswith("am") or candidate.endswith("pm"):
        digits = candidate[:-2].replace(":", "")
        return digits.isdigit()
    return False


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


def _llm_classify(
    *,
    text: str,
    normalized_text: str,
    pending_calendar: bool,
    pending_gmail: bool,
) -> ConfirmationIntent | None:
    if not settings.confirmation_intent_llm_enabled:
        return None
    response = call_json_chat_completion(
        provider=settings.planner_llm_provider,
        model=settings.planner_llm_model,
        api_key=settings.planner_llm_api_key,
        timeout_seconds=settings.confirmation_intent_llm_timeout_seconds,
        api_base_url=settings.planner_llm_api_base_url,
        system_prompt=_confirmation_classifier_prompt(),
        user_prompt=_confirmation_classifier_user_prompt(
            text=text,
            normalized_text=normalized_text,
            pending_calendar=pending_calendar,
            pending_gmail=pending_gmail,
        ),
        max_tokens=90,
        temperature=0.0,
    )
    if response.error is not None or not response.data:
        return None
    raw_intent = str(response.data.get("intent") or "").strip().lower()
    intent = raw_intent if raw_intent in _VALID_INTENTS else "unknown"
    confidence = _coerce_confidence(response.data.get("confidence"))
    reason = str(response.data.get("reason") or "").strip() or "llm_confirmation_classifier"
    if (
        intent in {"confirm", "cancel", "pause", "status", "edit"}
        and confidence < settings.confirmation_intent_llm_min_confidence
    ):
        return None
    return ConfirmationIntent(
        intent=intent,
        confidence=confidence,
        source="llm",
        normalized_text=normalized_text,
        reason=reason,
    )


def _confirmation_classifier_prompt() -> str:
    return (
        "You classify short user replies for pending confirmation actions.\n"
        "Return strict JSON only.\n"
        "Allowed intents: confirm, cancel, pause, status, edit, unknown.\n"
        "Use status only for Gmail pending-send status checks.\n"
        "Use edit for calendar draft modifications.\n"
        "Do not over-trigger confirm/cancel from weak language.\n"
        "Schema: {\"intent\":\"...\",\"confidence\":0.0,\"reason\":\"short\"}"
    )


def _confirmation_classifier_user_prompt(
    *,
    text: str,
    normalized_text: str,
    pending_calendar: bool,
    pending_gmail: bool,
) -> str:
    return (
        f"User reply:\n{text.strip()}\n\n"
        f"Normalized reply:\n{normalized_text}\n\n"
        f"Pending calendar draft: {'yes' if pending_calendar else 'no'}\n"
        f"Pending gmail send: {'yes' if pending_gmail else 'no'}\n\n"
        "Return JSON only."
    )
