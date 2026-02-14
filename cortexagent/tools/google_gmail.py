from __future__ import annotations

import base64
from email.message import EmailMessage
from email.utils import formatdate
import json
import re
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

from .base import Tool, ToolContext, ToolResult, ToolResultItem


class GoogleGmailTool(Tool):
    name = "google_gmail"
    GMAIL_THREADS_URL = "https://gmail.googleapis.com/gmail/v1/users/me/threads"
    GMAIL_MESSAGES_URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages"
    GMAIL_DRAFTS_URL = "https://gmail.googleapis.com/gmail/v1/users/me/drafts"
    GMAIL_SEND_URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"

    def run(self, context: ToolContext) -> ToolResult:
        tool_meta = context.tool_meta or {}
        access_token = tool_meta.get("access_token")
        if not isinstance(access_token, str) or not access_token.strip():
            raise RuntimeError("Google account is not connected. Please connect Google first.")

        token = access_token.strip()
        user_text = (context.user_text or "").strip()
        if not user_text:
            return ToolResult(tool_name=self.name, query="", items=[])

        allowed_domains = _parse_allowed_domains(tool_meta.get("allowed_recipient_domains"))

        if _is_send_intent(user_text):
            if not _has_explicit_send_confirmation(user_text):
                pending = self._build_send_confirmation_item(
                    access_token=token,
                    user_text=user_text,
                    allowed_domains=allowed_domains,
                )
                return ToolResult(tool_name=self.name, query=user_text, items=[pending])
            sent = self._send_message(
                access_token=token,
                user_text=_strip_confirmation_prefix(user_text),
                allowed_domains=allowed_domains,
            )
            return ToolResult(tool_name=self.name, query=user_text, items=[sent])

        if _is_draft_reply_intent(user_text):
            drafted = self._draft_reply(access_token=token, user_text=user_text)
            return ToolResult(tool_name=self.name, query=user_text, items=[drafted])

        if _is_read_message_intent(user_text):
            read_item = self._read_message(access_token=token, user_text=user_text)
            return ToolResult(tool_name=self.name, query=user_text, items=[read_item])

        max_results = _extract_max_results(user_text=user_text, default=5, minimum=1, maximum=15)
        threads = self._list_recent_threads(access_token=token, max_results=max_results)
        return ToolResult(tool_name=self.name, query=user_text, items=threads)

    def _list_recent_threads(self, access_token: str, max_results: int) -> list[ToolResultItem]:
        params = urlparse.urlencode({"maxResults": str(max_results), "q": "in:inbox"})
        url = f"{self.GMAIL_THREADS_URL}?{params}"
        payload = _api_request_json(
            url=url,
            method="GET",
            access_token=access_token,
            timeout=8,
            service_name="Gmail",
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
            details = self._get_thread_metadata(access_token=access_token, thread_id=thread_id)
            subject = details.get("subject") or "(no subject)"
            sender = details.get("from") or "Unknown sender"
            snippet = details.get("snippet") or ""
            out.append(
                ToolResultItem(
                    title=f"Thread {thread_id} | {subject}",
                    url=f"https://mail.google.com/mail/u/0/#inbox/{thread_id}",
                    snippet=f"From: {sender} | {snippet}".strip(" |"),
                )
            )
        return out

    def _read_message(self, access_token: str, user_text: str) -> ToolResultItem:
        thread_id = _extract_thread_id(user_text)
        if thread_id:
            details = self._get_thread_metadata(
                access_token=access_token,
                thread_id=thread_id,
                include_body=True,
            )
            return self._build_read_item(thread_id=thread_id, details=details)

        newest_threads = self._list_recent_threads(access_token=access_token, max_results=1)
        if not newest_threads:
            raise RuntimeError("I could not find any inbox threads to read.")
        newest_title = newest_threads[0].title
        inferred = _extract_thread_id(newest_title)
        if not inferred:
            raise RuntimeError("Could not resolve thread id for the newest email.")
        details = self._get_thread_metadata(
            access_token=access_token,
            thread_id=inferred,
            include_body=True,
        )
        return self._build_read_item(thread_id=inferred, details=details)

    def _build_read_item(self, thread_id: str, details: dict[str, str]) -> ToolResultItem:
        subject = details.get("subject") or "(no subject)"
        sender = details.get("from") or "Unknown sender"
        body = details.get("body") or details.get("snippet") or ""
        filtered_body, flagged = _sanitize_email_text(body)
        snippet = f"From: {sender}\n\n{filtered_body}".strip()
        if flagged:
            snippet += (
                "\n\n[Security] Potential prompt-injection lines were filtered from this email."
            )
        return ToolResultItem(
            title=f"Message from thread {thread_id} | {subject}",
            url=f"https://mail.google.com/mail/u/0/#inbox/{thread_id}",
            snippet=snippet,
        )

    def _draft_reply(self, access_token: str, user_text: str) -> ToolResultItem:
        thread_id = _extract_thread_id(user_text)
        if not thread_id:
            raise RuntimeError(
                "Please include a Gmail thread id to draft a reply. Example: "
                "'draft reply to thread 189abc: Thanks, I can do 3 PM.'"
            )
        body = _extract_reply_body(user_text)
        if not body:
            raise RuntimeError(
                "Please include the reply body after a colon. Example: "
                "'draft reply to thread 189abc: Thanks, that works for me.'"
            )
        details = self._get_thread_metadata(
            access_token=access_token,
            thread_id=thread_id,
            include_body=False,
            include_message_ids=True,
        )
        to_addr = details.get("reply_to") or details.get("from_email") or ""
        if not to_addr:
            raise RuntimeError("Could not infer recipient from that thread.")
        subject = details.get("subject") or ""
        message_id = details.get("message_id") or ""

        raw = _build_reply_rfc822_raw(
            to_addr=to_addr,
            subject=subject,
            body=body,
            message_id=message_id,
        )
        payload = {
            "message": {
                "threadId": thread_id,
                "raw": raw,
            }
        }
        created = _api_request_json(
            url=self.GMAIL_DRAFTS_URL,
            method="POST",
            access_token=access_token,
            timeout=8,
            service_name="Gmail",
            body=payload,
        )
        draft_id = str(created.get("id") or "").strip() if isinstance(created, dict) else ""
        if not draft_id:
            raise RuntimeError("Gmail returned an unexpected draft payload.")
        safe_preview, flagged = _sanitize_email_text(body)
        preview_note = (
            "\n[Security] Prompt-injection-like lines were removed from the draft preview."
            if flagged
            else ""
        )
        return ToolResultItem(
            title=f"[Drafted] Thread {thread_id}",
            url=f"https://mail.google.com/mail/u/0/#drafts?compose={draft_id}",
            snippet=(
                f"Draft id: {draft_id} | To: {to_addr} | Subject: {_ensure_reply_subject(subject)}\n"
                f"Preview: {safe_preview[:220]}{preview_note}"
            ),
        )

    def _build_send_confirmation_item(
        self,
        access_token: str,
        user_text: str,
        allowed_domains: set[str],
    ) -> ToolResultItem:
        draft_id = _extract_draft_id(user_text)
        if not draft_id:
            raise RuntimeError(
                "Please specify which draft to send. Example: 'send draft r-123abc'."
            )
        draft_details = self._get_draft(access_token=access_token, draft_id=draft_id)
        to_addr = draft_details.get("to") or ""
        subject = draft_details.get("subject") or "(no subject)"
        _enforce_allowed_recipient_domains(to_addr=to_addr, allowed_domains=allowed_domains)
        return ToolResultItem(
            title="Send confirmation required",
            url=f"https://mail.google.com/mail/u/0/#drafts?compose={draft_id}",
            snippet=(
                "I am ready to send this draft:\n"
                f"- Draft ID: {draft_id}\n"
                f"- To: {to_addr or 'unknown'}\n"
                f"- Subject: {subject}\n"
                "Reply with 'confirm send draft "
                f"{draft_id}' to send, or 'cancel' to stop."
            ),
        )

    def _send_message(
        self,
        access_token: str,
        user_text: str,
        allowed_domains: set[str],
    ) -> ToolResultItem:
        draft_id = _extract_draft_id(user_text)
        if not draft_id:
            raise RuntimeError(
                "Please specify which draft to send. Example: 'confirm send draft r-123abc'."
            )
        draft_details = self._get_draft(access_token=access_token, draft_id=draft_id)
        to_addr = draft_details.get("to") or ""
        _enforce_allowed_recipient_domains(to_addr=to_addr, allowed_domains=allowed_domains)
        payload = {"draftId": draft_id}
        sent = _api_request_json(
            url=self.GMAIL_SEND_URL,
            method="POST",
            access_token=access_token,
            timeout=8,
            service_name="Gmail",
            body=payload,
        )
        message_id = str(sent.get("id") or "").strip() if isinstance(sent, dict) else ""
        thread_id = str(sent.get("threadId") or "").strip() if isinstance(sent, dict) else ""
        if not message_id:
            raise RuntimeError("Gmail returned an unexpected send payload.")
        thread_link = (
            f"https://mail.google.com/mail/u/0/#inbox/{thread_id}"
            if thread_id
            else "https://mail.google.com/mail/u/0/#inbox"
        )
        return ToolResultItem(
            title=f"[Sent] Draft {draft_id}",
            url=thread_link,
            snippet=f"Message id: {message_id} | To: {to_addr or 'unknown'}",
        )

    def _get_draft(self, access_token: str, draft_id: str) -> dict[str, str]:
        url = f"{self.GMAIL_DRAFTS_URL}/{draft_id}?format=metadata"
        payload = _api_request_json(
            url=url,
            method="GET",
            access_token=access_token,
            timeout=8,
            service_name="Gmail",
        )
        if not isinstance(payload, dict):
            raise RuntimeError("Gmail returned an unexpected draft payload.")
        message = payload.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("Could not resolve draft message metadata.")
        headers = _headers_to_map(message)
        return {
            "to": headers.get("to", ""),
            "subject": headers.get("subject", ""),
        }

    def _get_thread_metadata(
        self,
        access_token: str,
        thread_id: str,
        include_body: bool = False,
        include_message_ids: bool = False,
    ) -> dict[str, str]:
        format_name = "full" if include_body else "metadata"
        url = f"{self.GMAIL_THREADS_URL}/{thread_id}?format={format_name}"
        payload = _api_request_json(
            url=url,
            method="GET",
            access_token=access_token,
            timeout=8,
            service_name="Gmail",
        )
        if not isinstance(payload, dict):
            return {}
        snippet = str(payload.get("snippet") or "").strip()
        messages = payload.get("messages", [])
        if not isinstance(messages, list) or not messages:
            return {"snippet": snippet}

        latest = messages[-1] if isinstance(messages[-1], dict) else {}
        if not isinstance(latest, dict):
            return {"snippet": snippet}
        headers = _headers_to_map(latest)
        from_header = headers.get("from", "")
        reply_to_header = headers.get("reply-to", "")
        message_id = headers.get("message-id", "")
        subject = headers.get("subject", "")

        body_text = ""
        if include_body:
            body_text = _extract_message_text(latest)

        out = {
            "subject": subject,
            "from": from_header,
            "from_email": _extract_first_email(from_header),
            "reply_to": _extract_first_email(reply_to_header),
            "snippet": snippet,
            "body": body_text,
        }
        if include_message_ids:
            out["message_id"] = message_id
        return out


def _extract_max_results(user_text: str, default: int, minimum: int, maximum: int) -> int:
    match = re.search(r"\b(\d{1,3})\b", user_text)
    if not match:
        return default
    try:
        value = int(match.group(1))
    except ValueError:
        return default
    return min(maximum, max(minimum, value))


def _is_read_message_intent(text: str) -> bool:
    lowered = text.strip().lower()
    return bool(
        re.search(r"\b(read|open|show)\b", lowered)
        and re.search(r"\b(email|message|thread|latest)\b", lowered)
    )


def _is_draft_reply_intent(text: str) -> bool:
    lowered = text.strip().lower()
    return bool(
        re.search(r"\b(draft|compose|write)\b", lowered)
        and re.search(r"\b(reply|response|email)\b", lowered)
    )


def _is_send_intent(text: str) -> bool:
    lowered = text.strip().lower()
    return bool(re.search(r"\b(send)\b", lowered) and re.search(r"\bdraft\b", lowered))


def _has_explicit_send_confirmation(text: str) -> bool:
    lowered = text.strip().lower()
    return bool(
        lowered.startswith("confirm send")
        or lowered.startswith("confirm: send")
        or "yes, send draft" in lowered
        or "yes send draft" in lowered
    )


def _strip_confirmation_prefix(text: str) -> str:
    cleaned = text.strip()
    lowered = cleaned.lower()
    if lowered.startswith("confirm:"):
        return cleaned[len("confirm:") :].strip()
    if lowered.startswith("confirm "):
        return cleaned[len("confirm ") :].strip()
    return cleaned


def _extract_thread_id(text: str) -> str | None:
    match = re.search(r"\bthread\s+([a-z0-9_-]{6,})\b", text.strip(), flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _extract_draft_id(text: str) -> str | None:
    match = re.search(r"\bdraft\s+([a-z0-9_-]{4,})\b", text.strip(), flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _extract_reply_body(text: str) -> str:
    parts = text.split(":", 1)
    if len(parts) < 2:
        return ""
    return parts[1].strip()


def _headers_to_map(message_payload: dict[str, object]) -> dict[str, str]:
    payload = message_payload.get("payload")
    if not isinstance(payload, dict):
        return {}
    raw_headers = payload.get("headers", [])
    if not isinstance(raw_headers, list):
        return {}
    out: dict[str, str] = {}
    for row in raw_headers:
        if not isinstance(row, dict):
            continue
        key = str(row.get("name") or "").strip().lower()
        value = str(row.get("value") or "").strip()
        if key:
            out[key] = value
    return out


def _extract_message_text(message_payload: dict[str, object]) -> str:
    payload = message_payload.get("payload")
    if not isinstance(payload, dict):
        return ""

    candidate = _walk_payload_for_text_part(payload)
    if candidate:
        return candidate
    body = payload.get("body")
    if isinstance(body, dict):
        data = body.get("data")
        if isinstance(data, str):
            return _decode_base64url_to_text(data)
    return ""


def _walk_payload_for_text_part(node: dict[str, object]) -> str:
    mime_type = str(node.get("mimeType") or "").lower()
    body = node.get("body")
    if mime_type == "text/plain" and isinstance(body, dict):
        data = body.get("data")
        if isinstance(data, str) and data.strip():
            return _decode_base64url_to_text(data)

    parts = node.get("parts")
    if not isinstance(parts, list):
        return ""
    for part in parts:
        if not isinstance(part, dict):
            continue
        text = _walk_payload_for_text_part(part)
        if text:
            return text
    return ""


def _decode_base64url_to_text(raw: str) -> str:
    padded = raw + ("=" * (-len(raw) % 4))
    try:
        decoded = base64.urlsafe_b64decode(padded.encode("ascii"))
        return decoded.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _build_reply_rfc822_raw(
    to_addr: str,
    subject: str,
    body: str,
    message_id: str,
) -> str:
    msg = EmailMessage()
    msg["To"] = to_addr
    msg["Subject"] = _ensure_reply_subject(subject)
    msg["Date"] = formatdate(localtime=True)
    if message_id:
        msg["In-Reply-To"] = message_id
        msg["References"] = message_id
    msg.set_content(body)
    raw = msg.as_bytes()
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _ensure_reply_subject(subject: str) -> str:
    cleaned = (subject or "").strip()
    if not cleaned:
        return "Re: (no subject)"
    if cleaned.lower().startswith("re:"):
        return cleaned
    return f"Re: {cleaned}"


def _extract_first_email(value: str) -> str:
    lowered = (value or "").strip()
    match = re.search(r"([A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,})", lowered, re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).strip().lower()


def _parse_allowed_domains(raw: object) -> set[str]:
    if not isinstance(raw, list):
        return set()
    out: set[str] = set()
    for row in raw:
        if not isinstance(row, str):
            continue
        cleaned = row.strip().lower()
        if cleaned:
            out.add(cleaned)
    return out


def _enforce_allowed_recipient_domains(to_addr: str, allowed_domains: set[str]) -> None:
    if not to_addr:
        raise RuntimeError("Could not verify recipient domain for this draft.")
    if not allowed_domains:
        return
    domain = to_addr.split("@", 1)[1].lower() if "@" in to_addr else ""
    if domain in allowed_domains:
        return
    raise RuntimeError(
        "Recipient domain is not allowed by policy. "
        f"Recipient: {to_addr}. Allowed domains: {', '.join(sorted(allowed_domains))}"
    )


def _sanitize_email_text(text: str) -> tuple[str, bool]:
    if not text.strip():
        return ("", False)
    suspicious = [
        "ignore previous instructions",
        "ignore all previous instructions",
        "system prompt",
        "reveal your prompt",
        "developer message",
        "tool output",
        "api key",
        "password",
        "secret token",
    ]
    flagged = False
    safe_lines: list[str] = []
    for line in text.splitlines():
        lowered = line.strip().lower()
        if any(marker in lowered for marker in suspicious):
            flagged = True
            continue
        safe_lines.append(line)
    cleaned = "\n".join(safe_lines).strip()
    if len(cleaned) > 2000:
        cleaned = cleaned[:2000].rstrip() + "..."
    return (cleaned, flagged)


def _api_request_json(
    url: str,
    method: str,
    access_token: str,
    timeout: int,
    service_name: str,
    body: dict[str, object] | None = None,
) -> dict:
    encoded = None if body is None else json.dumps(body).encode("utf-8")
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urlrequest.Request(url, data=encoded, method=method, headers=headers)
    try:
        with urlrequest.urlopen(req, timeout=timeout) as res:
            payload = json.loads(res.read().decode("utf-8"))
    except urlerror.HTTPError as exc:
        if exc.code in {401, 403}:
            raise RuntimeError(
                f"{service_name} authorization failed. Please reconnect Google."
            )
        if exc.code == 404:
            raise RuntimeError(f"{service_name} could not find the requested resource.")
        raise RuntimeError(f"{service_name} API failed ({exc.code}).")
    except Exception as exc:
        raise RuntimeError(f"{service_name} API failed: {exc}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"{service_name} returned an unexpected payload.")
    return payload
