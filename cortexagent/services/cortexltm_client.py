import json
from urllib import error as urlerror
from urllib import request as urlrequest


class CortexLTMClient:
    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _headers(self, authorization: str | None = None) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        if authorization:
            headers["Authorization"] = authorization
        return headers

    def chat(
        self,
        thread_id: str,
        text: str,
        short_term_limit: int | None,
        authorization: str | None,
    ) -> str:
        url = f"{self.base_url}/v1/threads/{thread_id}/chat"
        payload: dict[str, object] = {"text": text}
        if short_term_limit is not None:
            payload["short_term_limit"] = short_term_limit

        body = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(
            url,
            data=body,
            headers=self._headers(authorization=authorization),
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=25) as res:
                return res.read().decode("utf-8")
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"{exc.code} {detail or 'CortexLTM chat failed'}")
        except Exception as exc:
            raise RuntimeError(f"CortexLTM chat request failed: {exc}")

    def add_event(
        self,
        thread_id: str,
        actor: str,
        content: str,
        meta: dict[str, object] | None,
        authorization: str | None,
    ) -> str:
        url = f"{self.base_url}/v1/threads/{thread_id}/events"
        payload = {
            "actor": actor,
            "content": content,
            "meta": meta or {},
        }
        body = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(
            url,
            data=body,
            headers=self._headers(authorization=authorization),
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=15) as res:
                data = json.loads(res.read().decode("utf-8"))
                return str(data.get("event_id", ""))
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"{exc.code} {detail or 'CortexLTM add_event failed'}")
        except Exception as exc:
            raise RuntimeError(f"CortexLTM add_event request failed: {exc}")
