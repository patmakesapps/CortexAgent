from dataclasses import dataclass
import json
from urllib import error as urlerror
from urllib import request as urlrequest


@dataclass(frozen=True)
class LLMJsonResponse:
    data: dict[str, object] | None
    error: str | None


def call_json_chat_completion(
    *,
    provider: str,
    model: str,
    api_key: str | None,
    timeout_seconds: int,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 320,
    temperature: float = 0.0,
    api_base_url: str | None = None,
) -> LLMJsonResponse:
    normalized_provider = (provider or "").strip().lower()
    if not model:
        return LLMJsonResponse(data=None, error="missing_model")
    if not api_key:
        return LLMJsonResponse(data=None, error="missing_api_key")

    endpoint = _resolve_chat_endpoint(
        provider=normalized_provider,
        api_base_url=api_base_url,
    )
    if endpoint is None:
        return LLMJsonResponse(data=None, error="unsupported_provider")

    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    body = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=max(1, timeout_seconds)) as res:
            response_payload = json.loads(res.read().decode("utf-8"))
    except TimeoutError:
        return LLMJsonResponse(data=None, error="timeout")
    except urlerror.HTTPError:
        return LLMJsonResponse(data=None, error="http_error")
    except urlerror.URLError:
        return LLMJsonResponse(data=None, error="network_error")
    except json.JSONDecodeError:
        return LLMJsonResponse(data=None, error="invalid_provider_payload")
    except Exception:
        return LLMJsonResponse(data=None, error="request_failed")

    try:
        content = (
            response_payload.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
    except Exception:
        return LLMJsonResponse(data=None, error="invalid_provider_payload")

    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            chunk = str(part.get("text", "")).strip()
            if chunk:
                chunks.append(chunk)
        content = "\n".join(chunks)

    if not isinstance(content, str):
        return LLMJsonResponse(data=None, error="invalid_provider_payload")

    parsed = _extract_json_object(content)
    if parsed is None:
        return LLMJsonResponse(data=None, error="invalid_json")
    return LLMJsonResponse(data=parsed, error=None)


def _resolve_chat_endpoint(provider: str, api_base_url: str | None) -> str | None:
    if provider == "groq":
        return "https://api.groq.com/openai/v1/chat/completions"
    if provider in {"openai", "openai_compatible"}:
        base = (api_base_url or "https://api.openai.com/v1").strip()
        if not base:
            return None
        return f"{base.rstrip('/')}/chat/completions"
    return None


def _extract_json_object(content: str) -> dict[str, object] | None:
    raw = (content or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3:
            raw = "\n".join(lines[1:-1]).strip()
    try:
        value = json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(raw[start : end + 1])
        except Exception:
            return None
    if not isinstance(value, dict):
        return None
    return value
