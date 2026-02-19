from __future__ import annotations

import base64
import json
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

from .base import Tool, ToolContext, ToolResult, ToolResultItem


class GoogleGmailTool(Tool):
    name = "google_gmail"
    THREADS_URL = "https://gmail.googleapis.com/gmail/v1/users/me/threads"
    DRAFTS_URL = "https://gmail.googleapis.com/gmail/v1/users/me/drafts"
    SEND_URL = "https://gmail.googleapis.com/gmail/v1/users/me/drafts/send"
    PRIMARY_INBOX_QUERY = (
        "in:inbox category:primary -category:social "
        "-category:promotions -category:updates -category:forums"
    )

    def run(self, context: ToolContext) -> ToolResult:
        tool_meta = context.tool_meta or {}
        access_token = str(tool_meta.get("access_token") or "").strip()
        if not access_token:
            raise RuntimeError(
                "Google account is not connected. Please connect Google first."
            )

        operation = str(tool_meta.get("operation") or "read").strip().lower()
        args = tool_meta.get("args") if isinstance(tool_meta.get("args"), dict) else {}
        max_results = self._coerce_max_results(tool_meta.get("max_results"), default=5)
        query = str((args or {}).get("query") or "").strip()

        if operation == "send":
            draft_id = str((args or {}).get("draft_id") or "").strip()
            if not draft_id:
                raise RuntimeError("Missing draft_id for Gmail send operation.")
            sent = self._send_draft(access_token=access_token, draft_id=draft_id)
            return ToolResult(
                tool_name=self.name, query=context.user_text, items=[sent]
            )

        if operation == "draft_new":
            to_addr = str((args or {}).get("to") or "").strip()
            subject = str((args or {}).get("subject") or "").strip()
            body = str((args or {}).get("body") or "").strip()
            if not to_addr or not body:
                raise RuntimeError("Missing required args for draft_new: to, body.")
            drafted = self._draft_new_email(
                access_token=access_token,
                to_addr=to_addr,
                subject=subject or "(no subject)",
                body=body,
            )
            return ToolResult(
                tool_name=self.name, query=context.user_text, items=[drafted]
            )

        if operation == "draft_reply":
            thread_id = str((args or {}).get("thread_id") or "").strip()
            body = str((args or {}).get("body") or "").strip()
            if not thread_id or not body:
                raise RuntimeError(
                    "Missing required args for draft_reply: thread_id, body."
                )
            drafted = self._draft_reply(
                access_token=access_token,
                thread_id=thread_id,
                body=body,
            )
            return ToolResult(
                tool_name=self.name, query=context.user_text, items=[drafted]
            )

        if operation in {"read_message", "read_thread"}:
            thread_id = str((args or {}).get("thread_id") or "").strip()
            if not thread_id:
                raise RuntimeError("Missing thread_id for read_message operation.")
            details = self._get_thread_details(
                access_token=access_token, thread_id=thread_id
            )
            item = self._build_thread_item(thread_id=thread_id, details=details)
            return ToolResult(
                tool_name=self.name, query=context.user_text, items=[item]
            )

        threads = self._list_recent_threads(
            access_token=access_token,
            max_results=max_results,
            inbox_query=self._normalize_primary_query(query),
        )
        return ToolResult(tool_name=self.name, query=context.user_text, items=threads)

    @staticmethod
    def _coerce_max_results(value: object, default: int) -> int:
        if isinstance(value, int):
            return max(1, min(value, 5))
        if isinstance(value, str):
            try:
                parsed = int(value.strip())
            except ValueError:
                return default
            return max(1, min(parsed, 5))
        return default

    @classmethod
    def _normalize_primary_query(cls, raw_query: str) -> str:
        base = cls.PRIMARY_INBOX_QUERY
        query = (raw_query or "").strip()
        if not query:
            return base
        return f"({query}) {base}"

    def _list_recent_threads(
        self,
        *,
        access_token: str,
        max_results: int,
        inbox_query: str,
    ) -> list[ToolResultItem]:
        params = urlparse.urlencode({"maxResults": str(max_results), "q": inbox_query})
        payload = _api_request_json(
            url=f"{self.THREADS_URL}?{params}",
            access_token=access_token,
            method="GET",
        )
        rows = payload.get("threads", []) if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            return []

        out: list[ToolResultItem] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            thread_id = str(row.get("id") or "").strip()
            if not thread_id:
                continue
            details = self._get_thread_details(
                access_token=access_token, thread_id=thread_id
            )
            out.append(self._build_thread_item(thread_id=thread_id, details=details))
        return out

    def _build_thread_item(
        self, *, thread_id: str, details: dict[str, str]
    ) -> ToolResultItem:
        subject = details.get("subject") or "(no subject)"
        sender = details.get("from") or "Unknown sender"
        snippet = details.get("body") or "(empty body)"
        return ToolResultItem(
            title=f"Thread {thread_id} | {subject}",
            url=f"https://mail.google.com/mail/u/0/#inbox/{thread_id}",
            snippet=f"From: {sender}\n\n{snippet}".strip(),
        )

    def _get_thread_details(
        self, *, access_token: str, thread_id: str
    ) -> dict[str, str]:
        payload = _api_request_json(
            url=f"{self.THREADS_URL}/{thread_id}?format=full",
            access_token=access_token,
            method="GET",
        )
        if not isinstance(payload, dict):
            return {"subject": "", "from": "", "body": ""}
        messages = payload.get("messages", [])
        if not isinstance(messages, list) or not messages:
            return {"subject": "", "from": "", "body": ""}
        latest = messages[-1] if isinstance(messages[-1], dict) else {}
        parsed = _extract_message_fields(latest)
        return parsed

    def _draft_reply(
        self, *, access_token: str, thread_id: str, body: str
    ) -> ToolResultItem:
        details = self._get_thread_details(
            access_token=access_token, thread_id=thread_id
        )
        to_addr = details.get("reply_to") or details.get("from_email") or ""
        if not to_addr:
            raise RuntimeError("Could not infer reply recipient from thread.")
        subject = details.get("subject") or "(no subject)"
        raw = _build_email_rfc822_raw(
            to_addr=to_addr, subject=_reply_subject(subject), body=body
        )
        payload = {
            "message": {
                "raw": raw,
                "threadId": thread_id,
            }
        }
        created = _api_request_json(
            url=self.DRAFTS_URL,
            access_token=access_token,
            method="POST",
            body=payload,
        )
        draft_id = (
            str(created.get("id") or "").strip() if isinstance(created, dict) else ""
        )
        if not draft_id:
            raise RuntimeError("Gmail returned an unexpected draft payload.")
        return ToolResultItem(
            title=f"[Drafted Reply] {thread_id}",
            url=f"https://mail.google.com/mail/u/0/#drafts?compose={draft_id}",
            snippet=f"To: {to_addr} | Subject: {_reply_subject(subject)}",
        )

    def _draft_new_email(
        self,
        *,
        access_token: str,
        to_addr: str,
        subject: str,
        body: str,
    ) -> ToolResultItem:
        raw = _build_email_rfc822_raw(to_addr=to_addr, subject=subject, body=body)
        payload = {"message": {"raw": raw}}
        created = _api_request_json(
            url=self.DRAFTS_URL,
            access_token=access_token,
            method="POST",
            body=payload,
        )
        draft_id = (
            str(created.get("id") or "").strip() if isinstance(created, dict) else ""
        )
        if not draft_id:
            raise RuntimeError("Gmail returned an unexpected draft payload.")
        return ToolResultItem(
            title="[Drafted] New email",
            url=f"https://mail.google.com/mail/u/0/#drafts?compose={draft_id}",
            snippet=f"To: {to_addr} | Subject: {subject}",
        )

    def _send_draft(self, *, access_token: str, draft_id: str) -> ToolResultItem:
        payload = _api_request_json(
            url=self.SEND_URL,
            access_token=access_token,
            method="POST",
            body={"id": draft_id},
        )
        thread_id = (
            str(payload.get("threadId") or "").strip()
            if isinstance(payload, dict)
            else ""
        )
        link = (
            f"https://mail.google.com/mail/u/0/#inbox/{thread_id}"
            if thread_id
            else "https://mail.google.com/mail/u/0/#inbox"
        )
        return ToolResultItem(
            title="[Sent] Draft delivered", url=link, snippet=f"Draft id: {draft_id}"
        )


def _api_request_json(
    *,
    url: str,
    access_token: str,
    method: str,
    body: dict | None = None,
) -> dict:
    data = None
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
    }
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urlrequest.Request(url=url, method=method, data=data, headers=headers)
    try:
        with urlrequest.urlopen(req, timeout=10) as res:
            raw = res.read().decode("utf-8")
            parsed = json.loads(raw) if raw else {}
            return parsed if isinstance(parsed, dict) else {}
    except urlerror.HTTPError as exc:
        if exc.code in {401, 403}:
            raise RuntimeError("Gmail authorization failed. Please reconnect Google.")
        raise RuntimeError(f"Gmail API failed ({exc.code}).")
    except Exception as exc:
        raise RuntimeError(f"Gmail API failed: {exc}")


def _extract_message_fields(message: dict) -> dict[str, str]:
    payload = message.get("payload")
    if not isinstance(payload, dict):
        return {"subject": "", "from": "", "from_email": "", "reply_to": "", "body": ""}
    headers = payload.get("headers")
    header_map: dict[str, str] = {}
    if isinstance(headers, list):
        for row in headers:
            if not isinstance(row, dict):
                continue
            key = str(row.get("name") or "").strip().lower()
            value = str(row.get("value") or "").strip()
            if key and value:
                header_map[key] = value
    subject = header_map.get("subject", "")
    from_value = header_map.get("from", "")
    reply_to = header_map.get("reply-to", "")
    body = _extract_body_text(payload)
    return {
        "subject": subject,
        "from": from_value,
        "from_email": _extract_email_address(from_value),
        "reply_to": _extract_email_address(reply_to),
        "body": body,
    }


def _extract_body_text(payload: dict) -> str:
    body = payload.get("body")
    if isinstance(body, dict):
        data = str(body.get("data") or "").strip()
        if data:
            return _decode_base64url(data)
    parts = payload.get("parts")
    if isinstance(parts, list):
        for part in parts:
            if not isinstance(part, dict):
                continue
            nested = _extract_body_text(part)
            if nested:
                return nested
    return ""


def _decode_base64url(raw: str) -> str:
    padded = raw + ("=" * (-len(raw) % 4))
    try:
        decoded = base64.urlsafe_b64decode(padded.encode("ascii"))
    except Exception:
        return ""
    return decoded.decode("utf-8", errors="replace").strip()


def _extract_email_address(value: str) -> str:
    if "<" in value and ">" in value:
        start = value.find("<")
        end = value.find(">", start + 1)
        if end > start:
            return value[start + 1 : end].strip()
    return value.strip()


def _reply_subject(subject: str) -> str:
    cleaned = subject.strip()
    if cleaned.lower().startswith("re:"):
        return cleaned
    if not cleaned:
        return "Re: (no subject)"
    return f"Re: {cleaned}"


def _build_email_rfc822_raw(*, to_addr: str, subject: str, body: str) -> str:
    message = (
        f"To: {to_addr}\r\n"
        f"Subject: {subject}\r\n"
        "Content-Type: text/plain; charset=UTF-8\r\n"
        "\r\n"
        f"{body}"
    )
    encoded = base64.urlsafe_b64encode(message.encode("utf-8")).decode("ascii")
    return encoded.rstrip("=")
