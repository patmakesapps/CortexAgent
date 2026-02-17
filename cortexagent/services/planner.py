from dataclasses import dataclass
import re

from cortexagent.config import settings
from cortexagent.services.llm_json_client import call_json_chat_completion

_ALLOWED_ACTIONS = {
    "google_gmail",
    "google_calendar",
    "google_drive",
    "web_search",
    "chat",
}
_TOOL_ACTIONS = {"google_gmail", "google_calendar", "google_drive", "web_search"}
_ALLOWED_OUTCOMES_BY_ACTION: dict[str, set[str]] = {
    "google_gmail": {"gmail_send", "gmail_read"},
    "google_calendar": {"calendar_write", "calendar_read"},
    "google_drive": {"drive_search"},
    "web_search": {"web_search"},
    "chat": {"chat"},
}
_WRITE_OUTCOMES = {"gmail_send", "calendar_write"}
_ACTION_ALIASES = {
    "gmail": "google_gmail",
    "calendar": "google_calendar",
    "drive": "google_drive",
    "search": "web_search",
    "web": "web_search",
}
_OUTCOME_ALIASES = {
    "send_email": "gmail_send",
    "email_send": "gmail_send",
    "gmail_send_email": "gmail_send",
    "gmail_list": "gmail_read",
    "calendar_create": "calendar_write",
    "calendar_add": "calendar_write",
    "calendar_update": "calendar_write",
    "calendar_list": "calendar_read",
    "drive_read": "drive_search",
    "search_web": "web_search",
}


@dataclass(frozen=True)
class PlannerStep:
    step_id: str
    action: str
    query: str
    expected_outcome: str
    requires_confirmation: bool
    depends_on: list[str]


@dataclass(frozen=True)
class PlannerDecision:
    steps: list[PlannerStep]
    metadata: dict[str, object]


def build_orchestration_plan(
    *,
    user_text: str,
    prior_user_text: str | None,
    prior_tool_action: str | None,
    minimum_tool_steps: int = 2,
    tool_intent_required: bool = False,
    tool_intent_hints: list[str] | None = None,
) -> PlannerDecision:
    provider = settings.planner_llm_provider
    model = settings.planner_llm_model
    if not settings.planner_llm_enabled:
        return PlannerDecision(
            steps=[],
            metadata=_metadata(
                planner_used=False,
                planner_attempted=False,
                provider=provider,
                model=model,
                planner_confidence=0.0,
                planner_reason="disabled",
                validation_result="disabled",
                fallback_reason="planner_disabled",
            ),
        )

    response = call_json_chat_completion(
        provider=provider,
        model=model,
        api_key=settings.planner_llm_api_key,
        timeout_seconds=settings.planner_llm_timeout_seconds,
        api_base_url=settings.planner_llm_api_base_url,
        system_prompt=_planner_system_prompt(max_steps=settings.planner_llm_max_steps),
        user_prompt=_planner_user_prompt(
            user_text=user_text,
            prior_user_text=prior_user_text,
            prior_tool_action=prior_tool_action,
            max_steps=settings.planner_llm_max_steps,
            tool_intent_required=tool_intent_required,
            tool_intent_hints=tool_intent_hints or [],
        ),
        max_tokens=520,
        temperature=0.0,
    )
    if response.error is not None:
        return PlannerDecision(
            steps=[],
            metadata=_metadata(
                planner_used=False,
                planner_attempted=True,
                provider=provider,
                model=model,
                planner_confidence=0.0,
                planner_reason="llm_request_failed",
                validation_result=response.error,
                fallback_reason=response.error,
            ),
        )

    payload = response.data or {}
    raw_confidence = payload.get("planner_confidence")
    confidence = _coerce_confidence(raw_confidence)
    reason = str(payload.get("reason") or "").strip() or "llm_planner"
    if confidence < settings.planner_llm_min_confidence:
        return PlannerDecision(
            steps=[],
            metadata=_metadata(
                planner_used=False,
                planner_attempted=True,
                provider=provider,
                model=model,
                planner_confidence=confidence,
                planner_reason=reason,
                validation_result="low_confidence",
                fallback_reason="low_confidence",
            ),
        )

    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list):
        return PlannerDecision(
            steps=[],
            metadata=_metadata(
                planner_used=False,
                planner_attempted=True,
                provider=provider,
                model=model,
                planner_confidence=confidence,
                planner_reason=reason,
                validation_result="invalid_schema",
                fallback_reason="invalid_schema",
            ),
        )
    if len(raw_steps) > settings.planner_llm_max_steps:
        return PlannerDecision(
            steps=[],
            metadata=_metadata(
                planner_used=False,
                planner_attempted=True,
                provider=provider,
                model=model,
                planner_confidence=confidence,
                planner_reason=reason,
                validation_result="max_steps_exceeded",
                fallback_reason="max_steps_exceeded",
            ),
        )

    validated_steps = _normalize_steps(raw_steps=raw_steps, user_text=user_text)
    if not validated_steps:
        return PlannerDecision(
            steps=[],
            metadata=_metadata(
                planner_used=False,
                planner_attempted=True,
                provider=provider,
                model=model,
                planner_confidence=confidence,
                planner_reason=reason,
                validation_result="invalid_steps",
                fallback_reason="invalid_steps",
            ),
        )

    actionable_steps = [step for step in validated_steps if step.action in _TOOL_ACTIONS]
    if len(actionable_steps) < max(1, minimum_tool_steps):
        return PlannerDecision(
            steps=[],
            metadata=_metadata(
                planner_used=False,
                planner_attempted=True,
                provider=provider,
                model=model,
                planner_confidence=confidence,
                planner_reason=reason,
                validation_result=f"insufficient_steps_min_{max(1, minimum_tool_steps)}",
                fallback_reason=f"insufficient_steps_min_{max(1, minimum_tool_steps)}",
            ),
        )

    return PlannerDecision(
        steps=actionable_steps,
        metadata=_metadata(
            planner_used=True,
            planner_attempted=True,
            provider=provider,
            model=model,
            planner_confidence=confidence,
            planner_reason=reason,
            validation_result="valid",
            fallback_reason=None,
        ),
    )


def _normalize_steps(raw_steps: list[object], user_text: str) -> list[PlannerStep]:
    normalized: list[PlannerStep] = []
    seen_ids: set[str] = set()
    id_order: list[str] = []
    for index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, dict):
            return []

        raw_action = str(raw_step.get("action") or "").strip().lower()
        action = _ACTION_ALIASES.get(raw_action, raw_action)
        if action not in _ALLOWED_ACTIONS:
            return []

        step_id = str(raw_step.get("step_id") or raw_step.get("id") or "").strip().lower()
        if not step_id:
            step_id = f"step_{index + 1}"
        if step_id in seen_ids:
            step_id = f"{step_id}_{index + 1}"
        seen_ids.add(step_id)
        id_order.append(step_id)

        raw_query = str(raw_step.get("query") or "").strip()
        query = raw_query or user_text.strip()
        if not query:
            return []

        raw_expected = str(raw_step.get("expected_outcome") or "").strip().lower()
        expected_outcome = _OUTCOME_ALIASES.get(raw_expected, raw_expected)
        if not expected_outcome:
            expected_outcome = _infer_expected_outcome(action=action, query=query)
        allowed_outcomes = _ALLOWED_OUTCOMES_BY_ACTION.get(action, set())
        if expected_outcome not in allowed_outcomes:
            return []

        raw_requires_confirmation = raw_step.get("requires_confirmation")
        if isinstance(raw_requires_confirmation, bool):
            requires_confirmation = raw_requires_confirmation
        else:
            requires_confirmation = expected_outcome in _WRITE_OUTCOMES
        if expected_outcome in _WRITE_OUTCOMES:
            requires_confirmation = True

        depends_on_raw = raw_step.get("depends_on")
        depends_on: list[str] = []
        if isinstance(depends_on_raw, list):
            for dep in depends_on_raw:
                if not isinstance(dep, str):
                    continue
                candidate = dep.strip().lower()
                if candidate:
                    depends_on.append(candidate)

        normalized.append(
            PlannerStep(
                step_id=step_id,
                action=action,
                query=query,
                expected_outcome=expected_outcome,
                requires_confirmation=requires_confirmation,
                depends_on=depends_on,
            )
        )

    known_ids = set(id_order)
    filtered_steps: list[PlannerStep] = []
    for step in normalized:
        filtered_dependencies = [dep for dep in step.depends_on if dep in known_ids and dep != step.step_id]
        filtered_steps.append(
            PlannerStep(
                step_id=step.step_id,
                action=step.action,
                query=step.query,
                expected_outcome=step.expected_outcome,
                requires_confirmation=step.requires_confirmation,
                depends_on=filtered_dependencies,
            )
        )
    return filtered_steps


def _infer_expected_outcome(action: str, query: str) -> str:
    lowered = re.sub(r"\s+", " ", (query or "").strip().lower())
    if action == "google_gmail":
        if re.search(r"\b(confirm\s+send|send\s+draft|send|compose|write|email)\b", lowered):
            return "gmail_send"
        return "gmail_read"
    if action == "google_calendar":
        has_write_verb = bool(
            re.search(
                r"\b(add|create|schedule|book|set up|put|reschedule|move|shift|update|change|remind)\b",
                lowered,
            )
        )
        has_event_ref = bool(re.search(r"\b(calendar|meeting|event|appointment|reminder)\b", lowered))
        if has_write_verb and has_event_ref:
            return "calendar_write"
        return "calendar_read"
    if action == "google_drive":
        return "drive_search"
    if action == "web_search":
        return "web_search"
    return "chat"


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


def _planner_system_prompt(max_steps: int) -> str:
    return (
        "You are an orchestration planner for a chat agent.\n"
        "Available actions:\n"
        "- google_gmail\n"
        "- google_calendar\n"
        "- google_drive\n"
        "- web_search\n"
        "- chat\n\n"
        "Rules:\n"
        f"- Return up to {max_steps} steps.\n"
        "- Use only listed actions.\n"
        "- Output strict JSON only.\n"
        "- Do not include markdown or prose outside JSON.\n"
        "- If user intent includes sensitive writes (gmail send/calendar write), mark requires_confirmation=true.\n"
        "- If the user intent clearly maps to available tools (gmail/calendar/drive/web), you must return tool actions, not chat.\n"
        "- Only return chat when the user intent does not reasonably map to any listed tool.\n"
        "- Keep each query concise and directly executable.\n\n"
        "Output schema:\n"
        "{\n"
        '  "steps":[{\n'
        '    "step_id":"step_1",\n'
        '    "action":"google_gmail|google_calendar|google_drive|web_search|chat",\n'
        '    "query":"string",\n'
        '    "expected_outcome":"gmail_send|gmail_read|calendar_write|calendar_read|drive_search|web_search|chat",\n'
        '    "requires_confirmation":true,\n'
        '    "depends_on":["step_1"]\n'
        "  }],\n"
        '  "planner_confidence":0.0,\n'
        '  "reason":"short explanation"\n'
        "}"
    )


def _planner_user_prompt(
    *,
    user_text: str,
    prior_user_text: str | None,
    prior_tool_action: str | None,
    max_steps: int,
    tool_intent_required: bool,
    tool_intent_hints: list[str],
) -> str:
    hints = ", ".join(tool_intent_hints) if tool_intent_hints else "none"
    required_line = "yes" if tool_intent_required else "no"
    return (
        f"Current user message:\n{(user_text or '').strip()}\n\n"
        f"Prior user message:\n{(prior_user_text or '').strip() or 'none'}\n\n"
        f"Prior routed tool action:\n{(prior_tool_action or '').strip() or 'none'}\n\n"
        f"Tool intent required for this turn: {required_line}\n"
        f"Tool intent hints: {hints}\n\n"
        f"Max steps: {max_steps}\n"
        "Return strict JSON only."
    )


def _metadata(
    *,
    planner_used: bool,
    planner_attempted: bool,
    provider: str,
    model: str,
    planner_confidence: float,
    planner_reason: str,
    validation_result: str,
    fallback_reason: str | None,
) -> dict[str, object]:
    return {
        "planner_used": planner_used,
        "planner_attempted": planner_attempted,
        "planner_provider": provider,
        "planner_model": model,
        "planner_confidence": planner_confidence,
        "planner_reason": planner_reason,
        "validation_result": validation_result,
        "fallback_reason": fallback_reason,
        "planner_source": "llm" if planner_used else "fallback",
    }
