from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import re
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

from .base import Tool, ToolContext, ToolResult, ToolResultItem


class GoogleCalendarTool(Tool):
    name = "google_calendar"
    CALENDAR_EVENTS_URL = "https://www.googleapis.com/calendar/v3/calendars/primary/events"
    CALENDAR_QUICK_ADD_URL = (
        "https://www.googleapis.com/calendar/v3/calendars/primary/events/quickAdd"
    )

    def run(self, context: ToolContext) -> ToolResult:
        tool_meta = context.tool_meta or {}
        access_token = tool_meta.get("access_token")
        if not isinstance(access_token, str) or not access_token.strip():
            raise RuntimeError("Google account is not connected. Please connect Google first.")

        limit = tool_meta.get("max_results")
        try:
            max_results = int(limit) if limit is not None else 7
        except Exception:
            max_results = 7
        max_results = min(20, max(1, max_results))

        token = access_token.strip()
        items: list[ToolResultItem] = []
        is_create_intent = _is_create_intent(context.user_text)
        has_confirmation = _has_explicit_write_confirmation(context.user_text)
        if is_create_intent or has_confirmation:
            if not has_confirmation:
                return ToolResult(
                    tool_name=self.name,
                    query=context.user_text.strip(),
                    items=[
                        ToolResultItem(
                            title="Confirmation required",
                            url="https://calendar.google.com/",
                            snippet=_build_confirmation_prompt(context.user_text),
                        )
                    ],
                )
            created_item = self._quick_add_event(
                access_token=token,
                event_text=_strip_confirmation_prefix(context.user_text),
            )
            items.append(created_item)
            max_results = max(0, max_results - 1)

        if max_results > 0:
            items.extend(
                self._list_upcoming_events(
                    access_token=token,
                    max_results=max_results,
                )
            )
        return ToolResult(tool_name=self.name, query=context.user_text.strip(), items=items)

    def _quick_add_event(self, access_token: str, event_text: str) -> ToolResultItem:
        params = urlparse.urlencode({"text": event_text})
        url = f"{self.CALENDAR_QUICK_ADD_URL}?{params}"
        req = urlrequest.Request(
            url,
            method="POST",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {access_token}",
            },
        )
        try:
            with urlrequest.urlopen(req, timeout=8) as res:
                payload = json.loads(res.read().decode("utf-8"))
        except urlerror.HTTPError as exc:
            if exc.code in {401, 403}:
                raise RuntimeError("Google Calendar authorization failed. Please reconnect Google.")
            if exc.code == 400:
                raise RuntimeError(
                    "I could not create that event. Please include a clear date and time."
                )
            raise RuntimeError(f"Google Calendar API failed ({exc.code}).")
        except Exception as exc:
            raise RuntimeError(f"Google Calendar API failed: {exc}")

        if not isinstance(payload, dict):
            raise RuntimeError("Google Calendar returned an unexpected create-event payload.")

        title = str(payload.get("summary") or "Untitled event").strip()
        start_label = _event_time_label(payload.get("start"))
        location = str(payload.get("location") or "").strip()
        snippet = "Created event"
        if start_label:
            snippet += f" | {start_label}"
        if location:
            snippet += f" | {location}"
        return ToolResultItem(
            title=f"[Created] {title}",
            url=str(payload.get("htmlLink") or "https://calendar.google.com/").strip(),
            snippet=snippet,
        )

    def _list_upcoming_events(
        self, access_token: str, max_results: int
    ) -> list[ToolResultItem]:
        now = datetime.now(timezone.utc).isoformat()
        params = urlparse.urlencode(
            {
                "maxResults": str(max_results),
                "singleEvents": "true",
                "orderBy": "startTime",
                "timeMin": now,
            }
        )
        url = f"{self.CALENDAR_EVENTS_URL}?{params}"
        req = urlrequest.Request(
            url,
            method="GET",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {access_token}",
            },
        )
        try:
            with urlrequest.urlopen(req, timeout=8) as res:
                payload = json.loads(res.read().decode("utf-8"))
        except urlerror.HTTPError as exc:
            if exc.code in {401, 403}:
                raise RuntimeError("Google Calendar authorization failed. Please reconnect Google.")
            raise RuntimeError(f"Google Calendar API failed ({exc.code}).")
        except Exception as exc:
            raise RuntimeError(f"Google Calendar API failed: {exc}")

        raw_items = payload.get("items", []) if isinstance(payload, dict) else []
        if not isinstance(raw_items, list):
            return []

        out: list[ToolResultItem] = []
        for row in raw_items:
            if not isinstance(row, dict):
                continue
            title = str(row.get("summary") or "Untitled event").strip()
            start = row.get("start")
            start_label = _event_time_label(start)
            location = str(row.get("location") or "").strip()
            snippet = f"{start_label}"
            if location:
                snippet += f" | {location}"
            out.append(
                ToolResultItem(
                    title=title,
                    url="https://calendar.google.com/",
                    snippet=snippet,
                )
            )
        return out


def _event_time_label(start: object) -> str:
    if not isinstance(start, dict):
        return "Time unavailable"
    date_time = start.get("dateTime")
    if isinstance(date_time, str) and date_time.strip():
        normalized = date_time.strip()
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
            return "Starts: " + parsed.strftime("%a, %b %d at %I:%M %p").replace(
                " 0", " "
            )
        except ValueError:
            return f"Starts: {date_time.strip()}"
    date_only = start.get("date")
    if isinstance(date_only, str) and date_only.strip():
        try:
            parsed = datetime.fromisoformat(date_only.strip())
            return (
                "Starts: "
                + parsed.strftime("%a, %b %d")
                + " (all day)"
            )
        except ValueError:
            return f"Starts: {date_only.strip()} (all day)"
    return "Time unavailable"


def _is_create_intent(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    create_terms = (
        "create event",
        "add event",
        "add meeting",
        "create meeting",
        "schedule meeting",
        "book meeting",
        "set up meeting",
        "put on my calendar",
        "add to my calendar",
    )
    if any(term in lowered for term in create_terms):
        return True
    has_write_verb = bool(
        re.search(r"\b(add|ad|create|schedule|book|set up|put)\b", lowered)
    )
    has_calendar_ref = "calendar" in lowered
    has_event_ref = bool(re.search(r"\b(meeting|event|appointment|call)\b", lowered))
    has_pronoun_ref = bool(re.search(r"\b(it|this|that)\b", lowered))
    if has_write_verb and has_calendar_ref and (has_event_ref or has_pronoun_ref):
        return True
    pattern = re.compile(
        r"\b(add|ad|create|schedule|book|set up)\b.*\b(meeting|event|appointment)\b"
    )
    reverse_pattern = re.compile(
        r"\b(meeting|event|appointment|call)\b.*\b(add|ad|create|schedule|book|set up|put)\b"
    )
    return bool(pattern.search(lowered) or reverse_pattern.search(lowered))


def _has_explicit_write_confirmation(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    normalized = re.sub(r"\s+", " ", lowered).strip()
    if normalized in {"confirm", "yes create", "go ahead and create", "proceed to create"}:
        return True
    if normalized.startswith("confirm:"):
        return True
    if re.match(
        r"^\s*confirm\s+(?:add|create|schedule|book|set up|put)\b",
        normalized,
    ):
        return True
    return bool(
        re.match(
            r"^\s*(?:yes|ok|okay)\s*,?\s*(?:add|create|schedule|book|proceed)\b",
            normalized,
        )
    )


def _strip_confirmation_prefix(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return cleaned
    lowered = cleaned.lower()
    if lowered.startswith("confirm:"):
        cleaned = cleaned[len("confirm:") :].strip()
    elif lowered.startswith("confirm "):
        cleaned = cleaned[len("confirm ") :].strip()

    # Help Google quickAdd parse natural event text instead of command-style prompts.
    cleaned = re.sub(
        r"^(?:please\s+)?(?:add|create|schedule|book|set up)\s+"
        r"(?:an?\s+)?(?:event|meeting|appointment|call)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    return cleaned


def _build_confirmation_prompt(text: str) -> str:
    cleaned = (text or "").strip()
    lowered = cleaned.lower()
    name_match = re.search(
        r"\bwith\s+([a-z][a-z\s'-]{1,50}?)(?=\s+(?:on|at|in|and|for|its|it'?s|please|add|create|schedule|book)\b|[,.!?]|$)",
        lowered,
    )
    subject = "Meeting"
    if name_match:
        person = name_match.group(1).strip(" .")
        subject = "Meeting with " + " ".join(part.capitalize() for part in person.split())

    day_match = re.search(
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today|tonight)\b",
        lowered,
    )
    concrete_date_match = re.search(
        r"\b(?:"
        r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
        r")\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b"
        r"|"
        r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b",
        lowered,
    )
    assumed_day = False
    assumed_day_note = ""
    if concrete_date_match:
        day = concrete_date_match.group(0).strip()
    elif day_match:
        day_token = day_match.group(1).lower()
        if day_token in _WEEKDAY_TO_INDEX:
            day = _closest_future_day_label(preferred_weekday=_WEEKDAY_TO_INDEX[day_token])
            assumed_day = True
            assumed_day_note = f"closest upcoming {day_token.capitalize()}"
        elif day_token in {"today", "tonight"}:
            day = datetime.now().astimezone().strftime("%A, %b %d, %Y")
        else:
            day = (datetime.now().astimezone() + timedelta(days=1)).strftime(
                "%A, %b %d, %Y"
            )
    else:
        day = _closest_future_day_label(preferred_weekday=None)
        assumed_day = True
        assumed_day_note = "closest upcoming day"

    time_match = re.search(r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b", lowered)
    assumed_time = False
    if time_match:
        time_value = time_match.group(1).upper().replace(" ", "")
    else:
        time_value = "09:00AM"
        assumed_time = True

    if time_value.endswith("AM") or time_value.endswith("PM"):
        if ":" in time_value:
            pass
        elif len(time_value) <= 4:
            time_value = time_value[:-2] + ":00" + time_value[-2:]
        time_value = time_value[:-2] + " " + time_value[-2:]

    location_value = ""
    location_match = re.search(
        r"\b(?:at|in)\s+([a-z][a-z0-9&'.,\-\s]{1,60}?)(?=$|[,.!?])",
        lowered,
    )
    if location_match:
        candidate = re.sub(r"\s+", " ", location_match.group(1)).strip(" .")
        if candidate and not re.match(r"^\d{1,2}(?::\d{2})?\s*(?:am|pm)$", candidate):
            location_value = " ".join(part.capitalize() for part in candidate.split())

    assumption_lines: list[str] = []
    if assumed_day:
        assumption_lines.append(
            f"I assumed the {assumed_day_note} in the future: {day}."
        )
    if assumed_time:
        assumption_lines.append(f"I assumed the time: {time_value}.")

    assumption_block = ""
    if assumption_lines:
        assumption_block = "\nAssumptions to confirm: " + " ".join(assumption_lines)

    location_line = f"- Location: {location_value}\n" if location_value else ""
    return (
        "I have this draft event:\n"
        f"- Title: {subject}\n"
        f"- Day: {day}\n"
        f"- Time: {time_value}\n"
        f"{location_line}"
        f"{assumption_block}\n"
        "Should I add this to Google Calendar? "
        "Reply with 'confirm' to proceed or 'cancel' to stop."
    )


_WEEKDAY_TO_INDEX = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def _closest_future_day_label(preferred_weekday: int | None) -> str:
    today = datetime.now().astimezone().date()
    if preferred_weekday is None:
        days_ahead = 1
    else:
        # Monday=0 ... Sunday=6
        days_ahead = (preferred_weekday - today.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
    target = today + timedelta(days=days_ahead)
    return target.strftime("%A, %b %d, %Y")
