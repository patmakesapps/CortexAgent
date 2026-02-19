from __future__ import annotations

from typing import Any

import requests


class CortexLtmClient:
    def __init__(self, *, base_url: str, api_key: str | None) -> None:
        self._base_url = (base_url or "").strip().rstrip("/")
        if not self._base_url:
            raise RuntimeError("CORTEXLTM_API_BASE_URL is required.")
        self._api_key = (api_key or "").strip() or None

    def chat(
        self,
        *,
        thread_id: str,
        text: str,
        short_term_limit: int | None,
        authorization: str | None,
    ) -> str:
        payload: dict[str, Any] = {"text": text}
        if isinstance(short_term_limit, int):
            payload["short_term_limit"] = short_term_limit
        response = requests.post(
            self._url(f"/v1/threads/{thread_id}/chat"),
            headers=self._headers(authorization=authorization),
            json=payload,
            timeout=30,
        )
        if not response.ok:
            raise RuntimeError(
                f"CortexLTM chat failed ({response.status_code}): {self._error_message(response)}"
            )
        return response.text.strip()

    def create_event(
        self,
        *,
        thread_id: str,
        actor: str,
        content: str,
        meta: dict[str, object] | None,
        authorization: str | None,
    ) -> str:
        response = requests.post(
            self._url(f"/v1/threads/{thread_id}/events"),
            headers=self._headers(authorization=authorization),
            json={
                "actor": actor,
                "content": content,
                "meta": meta or {},
            },
            timeout=15,
        )
        if not response.ok:
            raise RuntimeError(
                f"Failed to persist event ({response.status_code}): {self._error_message(response)}"
            )
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("CortexLTM event create returned unexpected payload.")
        event_id = payload.get("event_id")
        if not isinstance(event_id, str) or not event_id.strip():
            raise RuntimeError("CortexLTM event create missing event_id.")
        return event_id.strip()

    def build_memory_context(
        self,
        *,
        thread_id: str,
        latest_user_text: str,
        short_term_limit: int | None,
        authorization: str | None,
    ) -> list[dict[str, str]]:
        payload: dict[str, object] = {"latest_user_text": latest_user_text}
        if isinstance(short_term_limit, int):
            payload["short_term_limit"] = short_term_limit
        response = requests.post(
            self._url(f"/v1/threads/{thread_id}/memory-context"),
            headers=self._headers(authorization=authorization),
            json=payload,
            timeout=20,
        )
        if not response.ok:
            raise RuntimeError(
                f"CortexLTM memory-context failed ({response.status_code}): {self._error_message(response)}"
            )
        parsed = response.json()
        if not isinstance(parsed, dict):
            return []
        rows = parsed.get("messages")
        if not isinstance(rows, list):
            return []
        out: list[dict[str, str]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            role = row.get("role")
            content = row.get("content")
            if isinstance(role, str) and isinstance(content, str):
                out.append({"role": role, "content": content})
        return out

    def _headers(self, *, authorization: str | None) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["x-api-key"] = self._api_key
        if isinstance(authorization, str) and authorization.strip():
            headers["Authorization"] = authorization.strip()
        return headers

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    @staticmethod
    def _error_message(response: requests.Response) -> str:
        text = response.text.strip()
        if not text:
            return "request failed"
        return text[:500]
