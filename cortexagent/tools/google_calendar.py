from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

from .base import Tool, ToolContext, ToolResult, ToolResultItem


class GoogleCalendarTool(Tool):
    name = "google_calendar"
    EVENTS_URL = "https://www.googleapis.com/calendar/v3/calendars/primary/events"
    QUICK_ADD_URL = "https://www.googleapis.com/calendar/v3/calendars/primary/events/quickAdd"

    def run(self, context: ToolContext) -> ToolResult:
        tool_meta = context.tool_meta or {}
        access_token = str(tool_meta.get("access_token") or "").strip()
        if not access_token:
            raise RuntimeError("Google account is not connected. Please connect Google first.")

        operation = str(tool_meta.get("operation") or "read").strip().lower()
        args = tool_meta.get("args") if isinstance(tool_meta.get("args"), dict) else {}
        max_results = self._coerce_max_results(tool_meta.get("max_results"), default=8)

        if operation in {"create", "write"}:
            event_text = str((args or {}).get("event_text") or context.user_text or "").strip()
            if not event_text:
                raise RuntimeError("Missing event_text for calendar create operation.")
            created = self._quick_add_event(access_token=access_token, event_text=event_text)
            return ToolResult(tool_name=self.name, query=event_text, items=[created])

        events = self._list_upcoming_events(access_token=access_token, max_results=max_results)
        return ToolResult(tool_name=self.name, query=context.user_text, items=events)

    @staticmethod
    def _coerce_max_results(value: object, default: int) -> int:
        if isinstance(value, int):
            return max(1, min(value, 50))
        if isinstance(value, str):
            try:
                parsed = int(value.strip())
            except ValueError:
                return default
            return max(1, min(parsed, 50))
        return default

    def _quick_add_event(self, access_token: str, event_text: str) -> ToolResultItem:
        params = urlparse.urlencode({"text": event_text})
        url = f"{self.QUICK_ADD_URL}?{params}"
        req = urlrequest.Request(
            url,
            method="POST",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {access_token}",
            },
        )
        try:
            with urlrequest.urlopen(req, timeout=10) as res:
                payload = json.loads(res.read().decode("utf-8"))
        except urlerror.HTTPError as exc:
            if exc.code in {401, 403}:
                raise RuntimeError("Google Calendar authorization failed. Please reconnect Google.")
            raise RuntimeError(f"Google Calendar API failed ({exc.code}).")
        except Exception as exc:
            raise RuntimeError(f"Google Calendar API failed: {exc}")

        if not isinstance(payload, dict):
            raise RuntimeError("Google Calendar returned an unexpected create payload.")

        title = str(payload.get("summary") or "Event").strip()
        event_link = str(payload.get("htmlLink") or "https://calendar.google.com").strip()
        created = str(payload.get("created") or "").strip()
        snippet = "Created event."
        if created:
            snippet = f"Created event at {created}."
        return ToolResultItem(title=f"[Created] {title}", url=event_link, snippet=snippet)

    def _list_upcoming_events(self, access_token: str, max_results: int) -> list[ToolResultItem]:
        now = datetime.now(timezone.utc)
        end = now + timedelta(days=30)
        params = urlparse.urlencode(
            {
                "maxResults": str(max_results),
                "orderBy": "startTime",
                "singleEvents": "true",
                "timeMin": now.isoformat().replace("+00:00", "Z"),
                "timeMax": end.isoformat().replace("+00:00", "Z"),
            }
        )
        url = f"{self.EVENTS_URL}?{params}"
        req = urlrequest.Request(
            url,
            method="GET",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {access_token}",
            },
        )
        try:
            with urlrequest.urlopen(req, timeout=10) as res:
                payload = json.loads(res.read().decode("utf-8"))
        except urlerror.HTTPError as exc:
            if exc.code in {401, 403}:
                raise RuntimeError("Google Calendar authorization failed. Please reconnect Google.")
            raise RuntimeError(f"Google Calendar API failed ({exc.code}).")
        except Exception as exc:
            raise RuntimeError(f"Google Calendar API failed: {exc}")

        rows = payload.get("items", []) if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            return []

        out: list[ToolResultItem] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            title = str(row.get("summary") or "Untitled event").strip()
            start_value = self._extract_start(row)
            link = str(row.get("htmlLink") or "https://calendar.google.com").strip()
            snippet = f"Start: {start_value}" if start_value else "Upcoming event"
            out.append(ToolResultItem(title=title, url=link, snippet=snippet))
        return out

    @staticmethod
    def _extract_start(row: dict) -> str:
        start = row.get("start")
        if not isinstance(start, dict):
            return ""
        return str(start.get("dateTime") or start.get("date") or "").strip()
