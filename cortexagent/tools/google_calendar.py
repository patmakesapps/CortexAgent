from __future__ import annotations

from datetime import datetime, timezone
import json
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

from .base import Tool, ToolContext, ToolResult, ToolResultItem


class GoogleCalendarTool(Tool):
    name = "google_calendar"
    CALENDAR_EVENTS_URL = "https://www.googleapis.com/calendar/v3/calendars/primary/events"

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

        items = self._list_upcoming_events(
            access_token=access_token.strip(),
            max_results=max_results,
        )
        return ToolResult(tool_name=self.name, query=context.user_text.strip(), items=items)

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
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone()
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
