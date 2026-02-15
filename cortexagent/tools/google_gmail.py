from __future__ import annotations

import base64
from email.message import EmailMessage
from email.utils import formatdate
from html import unescape as html_unescape
import json
import re
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

from cortexagent.config import settings

from .base import Tool, ToolContext, ToolResult, ToolResultItem


EXCLUDED_GMAIL_INBOX_LABELS = {
    "CATEGORY_PROMOTIONS",
    "CATEGORY_SOCIAL",
    "CATEGORY_UPDATES",
    "CATEGORY_FORUMS",
    "SPAM",
}


class GoogleGmailTool(Tool):
    name = "google_gmail"
    GMAIL_THREADS_URL = "https://gmail.googleapis.com/gmail/v1/users/me/threads"
    GMAIL_MESSAGES_URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages"
    GMAIL_DRAFTS_URL = "https://gmail.googleapis.com/gmail/v1/users/me/drafts"
    GMAIL_SEND_URL = "https://gmail.googleapis.com/gmail/v1/users/me/drafts/send"

    def run(self, context: ToolContext) -> ToolResult:
        tool_meta = context.tool_meta or {}
        access_token = tool_meta.get("access_token")
        if not isinstance(access_token, str) or not access_token.strip():
            raise RuntimeError("Google account is not connected. Please connect Google first.")

        token = access_token.strip()
        user_text = (context.user_text or "").strip()
        if not user_text:
            return ToolResult(tool_name=self.name, query="", items=[])
        inbox_query = _build_inbox_query(user_text)

        allowed_domains = _parse_allowed_domains(tool_meta.get("allowed_recipient_domains"))
        parsed_new_email: tuple[str, str, str, bool] | None = None

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

        if _is_send_new_email_intent(user_text):
            parsed_new_email = _extract_new_email_fields_detailed(user_text)
            pending = self._build_send_confirmation_for_new_email(
                access_token=token,
                user_text=user_text,
                allowed_domains=allowed_domains,
                parsed_email=parsed_new_email,
            )
            return ToolResult(tool_name=self.name, query=user_text, items=[pending])

        if _is_compose_new_email_intent(user_text):
            if parsed_new_email is None:
                parsed_new_email = _extract_new_email_fields_detailed(user_text)
            drafted = self._draft_new_email(
                access_token=token,
                user_text=user_text,
                allowed_domains=allowed_domains,
                parsed_email=parsed_new_email,
            )
            return ToolResult(tool_name=self.name, query=user_text, items=[drafted])

        if _is_draft_reply_intent(user_text):
            drafted = self._draft_reply(access_token=token, user_text=user_text)
            return ToolResult(tool_name=self.name, query=user_text, items=[drafted])

        if _is_read_message_intent(user_text):
            read_item = self._read_message(
                access_token=token,
                user_text=user_text,
                inbox_query=inbox_query,
            )
            return ToolResult(tool_name=self.name, query=user_text, items=[read_item])

        max_results = _extract_max_results(user_text=user_text, default=5, minimum=1, maximum=15)
        threads = self._list_recent_threads(
            access_token=token,
            max_results=max_results,
            inbox_query=inbox_query,
        )
        return ToolResult(tool_name=self.name, query=user_text, items=threads)

    def _list_recent_threads(
        self,
        access_token: str,
        max_results: int,
        inbox_query: str,
    ) -> list[ToolResultItem]:
        fetch_limit = min(max(max_results * 3, max_results), 50)
        params = urlparse.urlencode({"maxResults": str(fetch_limit), "q": inbox_query})
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
            details = self._get_thread_metadata(
                access_token=access_token,
                thread_id=thread_id,
                include_body=True,
            )
            if _has_excluded_inbox_labels(details.get("label_ids") or ""):
                continue
            subject = details.get("subject") or "(no subject)"
            sender = details.get("from") or "Unknown sender"
            body = details.get("body") or details.get("snippet") or ""
            safe_body, flagged = _sanitize_email_text(body)
            snippet = f"From: {sender}\n\n{safe_body}".strip()
            if flagged:
                snippet += (
                    "\n\n[Security] Potential prompt-injection lines were filtered from this email."
                )
            out.append(
                ToolResultItem(
                    title=f"Thread {thread_id} | {subject}",
                    url=f"https://mail.google.com/mail/u/0/#inbox/{thread_id}",
                    snippet=snippet,
                )
            )
            if len(out) >= max_results:
                break
        return out

    def _read_message(self, access_token: str, user_text: str, inbox_query: str) -> ToolResultItem:
        thread_id = _extract_thread_id(user_text)
        if thread_id:
            details = self._get_thread_metadata(
                access_token=access_token,
                thread_id=thread_id,
                include_body=True,
            )
            return self._build_read_item(thread_id=thread_id, details=details)

        candidate_threads = self._list_recent_threads(
            access_token=access_token,
            max_results=8,
            inbox_query=inbox_query,
        )
        if not candidate_threads:
            raise RuntimeError("I could not find any inbox threads to read.")
        inferred = _select_thread_id_for_read_intent(
            user_text=user_text,
            candidates=candidate_threads,
        )
        if not inferred:
            newest_title = candidate_threads[0].title
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
                f"To: {to_addr} | Subject: {_ensure_reply_subject(subject)}\n"
                f"Preview: {safe_preview[:220]}{preview_note}"
            ),
        )

    def _draft_new_email(
        self,
        access_token: str,
        user_text: str,
        allowed_domains: set[str],
        parsed_email: tuple[str, str, str, bool] | None = None,
    ) -> ToolResultItem:
        to_addr, subject, body, llm_used = parsed_email or _extract_new_email_fields_detailed(
            user_text
        )
        _enforce_allowed_recipient_domains(to_addr=to_addr, allowed_domains=allowed_domains)
        raw = _build_new_email_rfc822_raw(to_addr=to_addr, subject=subject, body=body)
        payload = {"message": {"raw": raw}}
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
        composer_note = " | Composer: LLM-assisted" if llm_used else ""
        return ToolResultItem(
            title="[Drafted] New email",
            url=f"https://mail.google.com/mail/u/0/#drafts?compose={draft_id}",
            snippet=(
                f"To: {to_addr} | Subject: {subject}{composer_note}\n"
                f"Preview: {safe_preview[:220]}{preview_note}"
            ),
        )

    def _build_send_confirmation_for_new_email(
        self,
        access_token: str,
        user_text: str,
        allowed_domains: set[str],
        parsed_email: tuple[str, str, str, bool] | None = None,
    ) -> ToolResultItem:
        parsed = parsed_email or _extract_new_email_fields_detailed(user_text)
        drafted = self._draft_new_email(
            access_token=access_token,
            user_text=user_text,
            allowed_domains=allowed_domains,
            parsed_email=parsed,
        )
        draft_id = _extract_draft_id(drafted.snippet or "") or _extract_draft_id(drafted.url or "")
        if not draft_id:
            raise RuntimeError("Could not resolve the draft id for send confirmation.")
        to_addr, subject, body, llm_used = parsed
        safe_body, flagged = _sanitize_email_text(body)
        body_line = safe_body[:240] if safe_body else "(empty body)"
        if flagged:
            body_line += " [sanitized]"
        composer_line = "- Composer: LLM-assisted\n" if llm_used else ""
        return ToolResultItem(
            title="Send confirmation required",
            url=f"https://mail.google.com/mail/u/0/#drafts?compose={draft_id}",
            snippet=(
                "I am ready to send this draft:\n"
                f"- To: {to_addr or 'unknown'}\n"
                f"- Subject: {subject}\n"
                f"{composer_line}"
                f"- Body: {body_line}\n"
                "Reply with 'confirm' to send, or 'cancel' to stop."
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
        body = draft_details.get("body") or ""
        safe_body, flagged = _sanitize_email_text(body)
        body_line = safe_body[:240] if safe_body else "(empty body)"
        if flagged:
            body_line += " [sanitized]"
        _enforce_allowed_recipient_domains(to_addr=to_addr, allowed_domains=allowed_domains)
        return ToolResultItem(
            title="Send confirmation required",
            url=f"https://mail.google.com/mail/u/0/#drafts?compose={draft_id}",
            snippet=(
                "I am ready to send this draft:\n"
                f"- To: {to_addr or 'unknown'}\n"
                f"- Subject: {subject}\n"
                f"- Body: {body_line}\n"
                "Reply with 'confirm' to send, or 'cancel' to stop."
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
        payload = {"id": draft_id}
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
            title="[Sent] Email",
            url=thread_link,
            snippet=f"To: {to_addr or 'unknown'}",
        )

    def _get_draft(self, access_token: str, draft_id: str) -> dict[str, str]:
        url = f"{self.GMAIL_DRAFTS_URL}/{draft_id}?format=full"
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
        body_text = _extract_message_text(message)
        return {
            "to": headers.get("to", ""),
            "subject": headers.get("subject", ""),
            "body": body_text,
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
        label_ids_raw = latest.get("labelIds", [])
        label_ids: list[str] = []
        if isinstance(label_ids_raw, list):
            for label in label_ids_raw:
                if isinstance(label, str) and label.strip():
                    label_ids.append(label.strip())

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
            "label_ids": ",".join(label_ids),
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


def _build_inbox_query(user_text: str) -> str:
    _ = user_text
    return (
        "in:inbox category:primary "
        "-category:promotions -category:social -category:updates -category:forums"
    )


def _has_excluded_inbox_labels(raw_labels: str) -> bool:
    labels = {value.strip().upper() for value in raw_labels.split(",") if value.strip()}
    return bool(labels.intersection(EXCLUDED_GMAIL_INBOX_LABELS))


def _is_read_message_intent(text: str) -> bool:
    lowered = text.strip().lower()
    if re.search(
        r"\bwhat\s+does\b.*\b(email|message|thread)\b.*\b(say|says)\b",
        lowered,
    ):
        return True
    if re.search(r"\btell me\b.*\b(email|message|thread)\b.*\b(say|says)\b", lowered):
        return True
    if re.search(r"\bwhat did\b.*\b(email|message|thread)\b.*\bsay\b", lowered):
        return True
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


def _is_send_new_email_intent(text: str) -> bool:
    lowered = text.strip().lower()
    if not lowered:
        return False
    if not _contains_email_address(lowered):
        return False
    # "draft ..." should stay draft-only unless user also asks to send.
    if re.search(r"\bdraft\b", lowered) and not re.search(r"\b(send|sned|snd)\b", lowered):
        return False
    if re.search(r"\b(send|sned|snd)\b", lowered):
        return True
    # Support imperative phrasing like:
    # "email person@example.com and write ...", "compose an email to ...", "email ... and say ..."
    if re.search(r"\b(email|compose|write|tell)\b", lowered):
        return True
    if re.search(
        r"\bemail\s+(?:to\s+)?[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b",
        lowered,
        re.IGNORECASE,
    ):
        return True
    return False


def _is_compose_new_email_intent(text: str) -> bool:
    lowered = text.strip().lower()
    if not lowered:
        return False
    return bool(
        re.search(r"\b(draft|compose|write)\b", lowered)
        and _contains_email_address(lowered)
    )


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


def _select_thread_id_for_read_intent(
    user_text: str,
    candidates: list[ToolResultItem],
) -> str | None:
    hint = _extract_read_target_hint(user_text)
    if not hint:
        return None
    lowered_hint = hint.lower()
    for item in candidates:
        haystack = f"{item.title} {item.snippet}".lower()
        if lowered_hint in haystack:
            thread_id = _extract_thread_id(item.title)
            if thread_id:
                return thread_id
    hint_tokens = [token for token in re.split(r"[^a-z0-9]+", lowered_hint) if len(token) >= 4]
    if not hint_tokens:
        return None
    best_thread: str | None = None
    best_score = 0
    for item in candidates:
        haystack = f"{item.title} {item.snippet}".lower()
        score = sum(1 for token in hint_tokens if token in haystack)
        if score > best_score:
            thread_id = _extract_thread_id(item.title)
            if thread_id:
                best_thread = thread_id
                best_score = score
    return best_thread


def _extract_read_target_hint(user_text: str) -> str:
    lowered = re.sub(r"\s+", " ", (user_text or "").strip().lower())
    if not lowered:
        return ""
    patterns = [
        r"\bwhat does the (.+?) email say\b",
        r"\bwhat does (.+?) email say\b",
        r"\bwhat did the (.+?) email say\b",
        r"\btell me what the (.+?) email says\b",
        r"\btell me what (.+?) says\b",
        r"\bread the (.+?) email\b",
        r"\bopen the (.+?) email\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if not match:
            continue
        candidate = match.group(1).strip()
        candidate = re.sub(r"\b(from|about|thread|message|email)\b", " ", candidate)
        candidate = re.sub(r"\s+", " ", candidate).strip(" .,!?:;")
        if candidate:
            return candidate
    return ""


def _extract_draft_id(text: str) -> str | None:
    cleaned = text.strip()
    match = re.search(
        r"\bdraft(?:\s+id)?\s*[:#]?\s*([a-z0-9_-]{3,})\b",
        cleaned,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    compose_match = re.search(r"\bcompose=([a-z0-9_-]{3,})\b", cleaned, flags=re.IGNORECASE)
    if compose_match:
        return compose_match.group(1).strip()
    return None


def _extract_reply_body(text: str) -> str:
    parts = text.split(":", 1)
    if len(parts) < 2:
        return ""
    return parts[1].strip()


def _extract_new_email_fields(text: str) -> tuple[str, str, str]:
    to_addr, subject, body, _llm_used = _extract_new_email_fields_detailed(text)
    return to_addr, subject, body


def _extract_new_email_fields_detailed(text: str) -> tuple[str, str, str, bool]:
    cleaned = _normalize_quote_chars(text)
    to_addr = _extract_first_email(cleaned)
    if not to_addr:
        raise RuntimeError("Please include a recipient email address, e.g. 'to name@example.com'.")

    subject = _extract_subject_text(cleaned)
    body = _extract_body_text(cleaned, known_subject=subject)
    if not body:
        body = _extract_freeform_body_text(cleaned, known_subject=subject)
    body = _trim_non_email_followup_fragments(body)
    body = _normalize_extracted_field(body, max_len=5000)
    if not body:
        raise RuntimeError(
            "Please tell me what the email should say, e.g. 'say \"Thanks for your time.\"'."
        )
    subject = _normalize_extracted_field(subject, max_len=300)
    if not subject:
        subject = _infer_subject_text(cleaned, body)
    subject = _normalize_extracted_field(subject, max_len=300) or "Quick note"

    llm_used = False
    should_use_composer = _should_use_llm_email_composer(cleaned, subject=subject, body=body)
    if should_use_composer:
        llm_result = _llm_compose_email_fields(
            user_text=cleaned,
            fallback_to=to_addr,
            fallback_subject=subject,
            fallback_body=body,
        )
        if llm_result is not None:
            llm_to, llm_subject, llm_body = llm_result
            to_addr = llm_to or to_addr
            subject = _normalize_extracted_field(llm_subject, max_len=300) or subject
            body = _normalize_extracted_field(llm_body, max_len=5000) or body
            llm_used = True
        else:
            subject, body = _compose_email_fields_fallback(
                user_text=cleaned,
                fallback_subject=subject,
                fallback_body=body,
            )

    if _looks_like_ambiguous_new_email_parse(body=body, subject=subject):
        raise RuntimeError(
            "I couldn’t confidently parse the email body. "
            "Please provide it as: say \"...\" (and optional subject \"...\")."
        )
    return to_addr, subject, body, llm_used


def _should_use_llm_email_composer(text: str, subject: str, body: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    explicit_compose_intent = bool(
        re.search(
            r"\b(compose|write|draft)\b.*\b(body|message|email)\b",
            lowered,
        )
    )
    body_quality_issue = bool(
        body.strip().lower().endswith("then")
        or body.strip().lower().endswith("and")
        or " then add " in body.lower()
        or " add it to my calendar" in body.lower()
        or " compose the body" in lowered
        or " in the email make the subject" in lowered
    )
    return explicit_compose_intent or body_quality_issue


def _llm_compose_email_fields(
    user_text: str,
    fallback_to: str,
    fallback_subject: str,
    fallback_body: str,
) -> tuple[str, str, str] | None:
    if not settings.groq_api_key:
        return None

    prompt = (
        "You are an email drafting assistant.\n"
        "Return strict JSON only with keys: to, subject, body.\n"
        "Rules:\n"
        "- Keep the recipient email exactly as requested.\n"
        "- Ignore non-email tasks such as calendar actions.\n"
        "- Subject should be concise and professional.\n"
        "- Body should be clear, complete, and user-ready.\n"
        "- Do not include markdown.\n"
    )
    payload = {
        "model": settings.router_llm_model,
        "temperature": 0.2,
        "max_tokens": 240,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"User request:\n{user_text}\n\n"
                    f"Fallback recipient: {fallback_to}\n"
                    f"Fallback subject: {fallback_subject}\n"
                    f"Fallback body: {fallback_body}\n"
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
        with urlrequest.urlopen(req, timeout=max(4, settings.router_llm_timeout_seconds)) as res:
            response_payload = json.loads(res.read().decode("utf-8"))
    except Exception:
        return None

    content = (
        response_payload.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    parsed = _extract_json_object(content)
    if not isinstance(parsed, dict):
        return None
    to_addr = _normalize_extracted_field(str(parsed.get("to") or fallback_to), max_len=320).lower()
    subject = _normalize_extracted_field(str(parsed.get("subject") or fallback_subject), max_len=300)
    email_body = _normalize_extracted_field(str(parsed.get("body") or fallback_body), max_len=5000)
    if not to_addr or "@" not in to_addr:
        to_addr = fallback_to
    if not subject:
        subject = fallback_subject or "Quick note"
    if not email_body:
        email_body = fallback_body
    if not email_body:
        return None
    return to_addr, subject, email_body


def _compose_email_fields_fallback(
    user_text: str,
    fallback_subject: str,
    fallback_body: str,
) -> tuple[str, str]:
    cleaned_user_text = _normalize_quote_chars(user_text or "")
    subject = _normalize_extracted_field(
        _trim_non_email_followup_fragments(fallback_subject),
        max_len=300,
    )
    body_intent = _normalize_extracted_field(
        _trim_non_email_followup_fragments(fallback_body),
        max_len=5000,
    )

    body_intent = re.sub(
        r"(?:\.\.\.|…|\bthen\b|\band then\b|\band\b)\s*$",
        "",
        body_intent,
        flags=re.IGNORECASE,
    ).strip(" .")
    if not body_intent:
        body_intent = "reaching out with a quick update"

    lowered_request = cleaned_user_text.lower()
    lowered_intent = body_intent.lower()

    if not subject:
        if re.search(r"\b(gratitude|grateful|appreciate|appreciation|thank)\b", lowered_request):
            subject = "Thank You for the Opportunity"
        else:
            subject = _infer_subject_from_body(body_intent) or "Quick note"
    subject = _normalize_extracted_field(subject, max_len=300) or "Quick note"
    mention_gratitude = bool(
        re.search(r"\b(gratitude|grateful|appreciate|appreciation|thank)\b", lowered_request)
        or re.search(r"\b(gratitude|grateful|appreciate|appreciation|thank)\b", lowered_intent)
    )
    normalized_intent = re.sub(r"\bexcepted\b", "accepted", body_intent, flags=re.IGNORECASE).strip()
    if lowered_intent.startswith("expressing my gratitude"):
        normalized_intent = (
            "I wanted to express my gratitude for the opportunity "
            "and share how excited I am to move forward."
        )

    if normalized_intent and normalized_intent[-1] not in ".!?":
        normalized_intent = f"{normalized_intent}."
    if normalized_intent:
        normalized_intent = normalized_intent[0].upper() + normalized_intent[1:]

    gratitude_sentence = (
        "Thank you again for the opportunity."
        if mention_gratitude
        else "Please let me know if you would like to discuss details or next steps."
    )

    composed = (
        "Hi,\n\n"
        f"{normalized_intent}\n\n"
        f"{gratitude_sentence}\n\n"
        "Best,\n"
        "Patrick"
    )
    return subject, composed


def _extract_json_object(content: str) -> dict[str, object] | None:
    raw = (content or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        rows = raw.splitlines()
        if len(rows) >= 3:
            raw = "\n".join(rows[1:-1]).strip()
    try:
        value = json.loads(raw)
    except Exception:
        return None
    if not isinstance(value, dict):
        return None
    return value


def _extract_subject_text(text: str) -> str:
    quoted = re.search(
        r"\bsubject(?:\s+to\s+be|\s+is)?\s*[:=\-]?\s*\"([^\"]{1,300})\"",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if quoted:
        return quoted.group(1).strip()
    inline = re.search(
        r"\bsubject(?:\s+to\s+be|\s+is)?\s*[:=\-]?\s*([^\n|]{1,300})",
        text,
        flags=re.IGNORECASE,
    )
    if not inline:
        return ""
    candidate = re.split(
        r"\b(?:and\s+the\s+email\s+body|and\s+body|body(?:\s+can\s+say|\s+is)?|message(?:\s+can\s+say|\s+is)?|say(?:ing)?)\b",
        inline.group(1),
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return candidate.strip(" -:;,.")


def _extract_body_text(text: str, known_subject: str = "") -> str:
    quoted = re.search(
        r"\b(?:email\s+)?body(?:\s+can\s+say|\s+should\s+say|\s+is)?\s*[:=\-]?\s*\"([^\"]{1,5000})\"",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if quoted:
        return quoted.group(1).strip()
    inline = re.search(
        r"\b(?:email\s+)?body(?:\s+can\s+say|\s+should\s+say|\s+is)?\s*[:=\-]?\s*(.+)$",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if inline:
        return inline.group(1).strip().strip('"')
    message_line = re.search(
        r"\bmessage(?:\s+can\s+say|\s+should\s+say|\s+is)?\s*[:=\-]?\s*(.+)$",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if message_line:
        return message_line.group(1).strip().strip('"')
    spoken_quoted = re.search(
        r"\b(?:saying|say)\b\s*[:=\-]?\s*\"([^\"]{1,5000})\"",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if spoken_quoted:
        return spoken_quoted.group(1).strip()
    spoken_inline = re.search(
        r"\b(?:saying|say)\b\s*[:=\-]?\s*(.+)$",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if spoken_inline:
        candidate = re.split(
            r"\bsubject(?:\s+to\s+be|\s+is)?\b",
            spoken_inline.group(1),
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        return _trim_non_email_followup_fragments(candidate.strip().strip('"'))
    tell_quoted = re.search(
        r"\btell\s+(?:him|her|them)\s*[:=\-]?\s*\"([^\"]{1,5000})\"",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if tell_quoted:
        return _trim_non_email_followup_fragments(tell_quoted.group(1).strip())
    tell_inline = re.search(
        r"\btell\s+(?:him|her|them)\s*[:=\-]?\s*(.+)$",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if tell_inline:
        return _trim_non_email_followup_fragments(tell_inline.group(1).strip().strip('"'))
    let_them_know = re.search(
        r"\blet\s+(?:him|her|them)\s+know\s*[:=\-]?\s*(.+)$",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if let_them_know:
        return _trim_non_email_followup_fragments(
            let_them_know.group(1).strip().strip('"')
        )
    quoted_chunks = re.findall(r"\"([^\"]{1,5000})\"", text, flags=re.DOTALL)
    for chunk in quoted_chunks:
        candidate = chunk.strip()
        if not candidate:
            continue
        if known_subject and candidate.lower() == known_subject.strip().lower():
            continue
        return _trim_non_email_followup_fragments(candidate)
    return ""


def _extract_freeform_body_text(text: str, known_subject: str) -> str:
    colon_style = re.search(
        r"\b(?:send|sned|snd|draft|compose|write)\b[^:\n]{0,220}:\s*(.+)$",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if colon_style:
        candidate = colon_style.group(1).strip().strip('"')
        if not _looks_like_non_body_fragment(candidate, known_subject):
            return _trim_non_email_followup_fragments(candidate)

    candidate = text
    candidate = re.sub(
        r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}",
        " ",
        candidate,
        flags=re.IGNORECASE,
    )
    candidate = re.sub(
        r"\bsubject(?:\s+to\s+be|\s+is)?\s*[:=\-]?\s*\"[^\"]{0,300}\"",
        " ",
        candidate,
        flags=re.IGNORECASE | re.DOTALL,
    )
    candidate = re.sub(
        r"\bsubject(?:\s+to\s+be|\s+is)?\s*[:=\-]?\s*([^\n|]{1,300})",
        " ",
        candidate,
        flags=re.IGNORECASE,
    )
    candidate = re.sub(
        r"\b(?:please|can you|could you|would you|i need you to)\b",
        " ",
        candidate,
        flags=re.IGNORECASE,
    )
    candidate = re.sub(r"\b(?:send|sned|snd|draft|compose|write)\b", " ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(
        r"\b(?:an?\s+)?(?:email|gmail|message)\b",
        " ",
        candidate,
        flags=re.IGNORECASE,
    )
    candidate = re.sub(r"\b(?:to|for|with|and)\b", " ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s+", " ", candidate).strip(" \t\r\n\"'.,;:-")
    if _looks_like_non_body_fragment(candidate, known_subject):
        return ""
    return _trim_non_email_followup_fragments(candidate)


def _trim_non_email_followup_fragments(value: str) -> str:
    candidate = (value or "").strip()
    if not candidate:
        return ""
    patterns = (
        r"\b(?:please\s+)?add\s+it\s+to\s+my\s+calendar\b.*$",
        r"\b(?:and|also)\s+(?:please\s+)?(?:add|put|schedule|create|book)\b.*\bcalendar\b.*$",
        r"\b(?:please\s+)?(?:add|put|schedule|create|book)\b.*\bcalendar\b.*$",
    )
    for pattern in patterns:
        candidate = re.sub(pattern, "", candidate, flags=re.IGNORECASE | re.DOTALL).strip()
    return candidate.strip(" .")


def _infer_subject_text(text: str, body: str) -> str:
    hint = _extract_subject_hint_text(text)
    if hint:
        return hint
    from_body = _infer_subject_from_body(body)
    if from_body:
        return from_body
    return "Quick note"


def _extract_subject_hint_text(text: str) -> str:
    quoted = re.search(
        r"\b(?:about|regarding|re)\s*[:=\-]?\s*\"([^\"]{1,300})\"",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if quoted:
        return quoted.group(1).strip()
    inline = re.search(
        r"\b(?:about|regarding|re)\s+(.+?)(?=\b(?:say|saying|body|message|subject)\b|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not inline:
        return ""
    return _normalize_extracted_field(inline.group(1), max_len=300)


def _infer_subject_from_body(body: str) -> str:
    cleaned = _normalize_extracted_field(body, max_len=300)
    if not cleaned:
        return ""
    words = cleaned.split()
    if not words:
        return ""
    if len(words) <= 8:
        return cleaned
    return " ".join(words[:8]).strip(" -:;,.")


def _normalize_extracted_field(value: str, max_len: int) -> str:
    cleaned = re.sub(r"\s+", " ", (value or "")).strip().strip('"').strip()
    cleaned = cleaned.strip(" -:;,.")
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip()
    return cleaned


def _looks_like_non_body_fragment(value: str, known_subject: str) -> bool:
    cleaned = _normalize_extracted_field(value, max_len=5000).lower()
    if not cleaned:
        return True
    if known_subject and cleaned == known_subject.strip().lower() and len(cleaned.split()) <= 6:
        return True
    if not re.search(r"[a-z0-9]", cleaned):
        return True
    if cleaned in {
        "send",
        "sned",
        "snd",
        "email",
        "gmail",
        "message",
        "draft",
        "compose",
        "write",
        "to",
        "for",
        "with",
    }:
        return True
    return False


def _looks_like_ambiguous_new_email_parse(body: str, subject: str) -> bool:
    normalized_body = _normalize_extracted_field(body, max_len=5000).lower()
    normalized_subject = _normalize_extracted_field(subject, max_len=300).lower()
    if not normalized_body:
        return True
    if normalized_body == normalized_subject and len(normalized_body.split()) <= 2:
        return True
    parser_residue = (
        normalized_body.startswith("subject ")
        or normalized_body.startswith("body ")
        or normalized_body.startswith("message ")
        or normalized_body in {"send", "email", "gmail", "draft"}
    )
    if parser_residue:
        return True
    return len(normalized_body) < 2


def _normalize_quote_chars(text: str) -> str:
    return (
        (text or "")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )


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
        text = decoded.decode("utf-8", errors="replace")
        return _html_to_text_if_needed(text)
    except Exception:
        return ""


def _html_to_text_if_needed(text: str) -> str:
    lowered = (text or "").strip().lower()
    if "<html" not in lowered and "<body" not in lowered and "<div" not in lowered:
        return text
    no_scripts = re.sub(r"(?is)<(script|style)\b.*?>.*?</\1>", " ", text)
    no_tags = re.sub(r"(?is)<[^>]+>", " ", no_scripts)
    collapsed = re.sub(r"\s+", " ", html_unescape(no_tags)).strip()
    return collapsed


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


def _build_new_email_rfc822_raw(to_addr: str, subject: str, body: str) -> str:
    msg = EmailMessage()
    msg["To"] = to_addr
    msg["Subject"] = (subject or "").strip() or "(no subject)"
    msg["Date"] = formatdate(localtime=True)
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


def _contains_email_address(value: str) -> bool:
    return bool(
        re.search(
            r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}",
            (value or "").strip(),
            re.IGNORECASE,
        )
    )


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
        error_detail = ""
        try:
            raw_error = exc.read().decode("utf-8", errors="replace")
            parsed_error = json.loads(raw_error)
            if isinstance(parsed_error, dict):
                nested = parsed_error.get("error")
                if isinstance(nested, dict):
                    message = str(nested.get("message") or "").strip()
                    if message:
                        error_detail = f": {message}"
        except Exception:
            error_detail = ""
        if exc.code in {401, 403}:
            raise RuntimeError(
                f"{service_name} authorization failed. Please reconnect Google."
            )
        if exc.code == 404:
            raise RuntimeError(f"{service_name} could not find the requested resource.")
        raise RuntimeError(f"{service_name} API failed ({exc.code}){error_detail}.")
    except Exception as exc:
        raise RuntimeError(f"{service_name} API failed: {exc}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"{service_name} returned an unexpected payload.")
    return payload
