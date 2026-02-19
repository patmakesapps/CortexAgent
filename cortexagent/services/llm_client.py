from __future__ import annotations

import json
from dataclasses import dataclass

import requests


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    provider: str
    model: str
    api_key: str
    timeout_seconds: int
    api_base_url: str | None = None


class OpenAICompatibleClient:
    def __init__(self, cfg: OpenAICompatibleConfig) -> None:
        provider = (cfg.provider or "").strip().lower()
        if provider not in {"groq", "openai", "openai_compatible"}:
            raise ValueError("provider must be one of: groq, openai, openai_compatible")

        api_key = (cfg.api_key or "").strip()
        if not api_key:
            raise RuntimeError("LLM API key is required.")

        self._provider = provider
        self._model = (cfg.model or "").strip()
        if not self._model:
            raise RuntimeError("LLM model is required.")

        self._api_key = api_key
        self._timeout_seconds = max(1, int(cfg.timeout_seconds))
        base = (cfg.api_base_url or "").strip()
        if not base:
            if provider == "groq":
                base = "https://api.groq.com/openai/v1"
            else:
                base = "https://api.openai.com/v1"
        self._base_url = base.rstrip("/")

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = requests.post(
            f"{self._base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self._timeout_seconds,
        )
        if not response.ok:
            detail = response.text.strip()
            raise RuntimeError(
                f"LLM completion failed ({response.status_code}): {detail[:400] or 'request failed'}"
            )
        body = response.json()
        if not isinstance(body, dict):
            raise RuntimeError("LLM completion returned unexpected payload.")
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("LLM completion returned no choices.")
        row = choices[0]
        if not isinstance(row, dict):
            raise RuntimeError("LLM completion returned malformed choice row.")
        message = row.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("LLM completion missing message payload.")
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        raise RuntimeError("LLM completion missing content.")


def extract_first_json_object(raw_text: str) -> dict[str, object] | None:
    text = (raw_text or "").strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None

    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
