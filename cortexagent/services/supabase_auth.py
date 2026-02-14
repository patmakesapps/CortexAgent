from __future__ import annotations

from typing import Any

import requests


def extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    raw = authorization.strip()
    if not raw:
        return None
    if not raw.lower().startswith("bearer "):
        return None
    token = raw[7:].strip()
    return token or None


def fetch_supabase_user_id(
    access_token: str,
    supabase_url: str | None,
    supabase_anon_key: str | None,
    timeout_seconds: int = 5,
) -> str:
    url = (supabase_url or "").strip().rstrip("/")
    anon_key = (supabase_anon_key or "").strip()
    if not url or not anon_key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_ANON_KEY are required to validate bearer auth."
        )

    response = requests.get(
        f"{url}/auth/v1/user",
        headers={
            "Authorization": f"Bearer {access_token}",
            "apikey": anon_key,
        },
        timeout=max(1, timeout_seconds),
    )
    if response.status_code == 401:
        raise ValueError("Invalid or expired access token.")
    if not response.ok:
        raise RuntimeError(
            f"Auth provider unavailable: HTTP {response.status_code} {response.text.strip()}"
        )

    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Auth provider returned unexpected payload.")
    user_id = payload.get("id")
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("Access token missing user id.")
    return user_id.strip()


def resolve_user_id_from_authorization(
    authorization: str | None,
    supabase_url: str | None,
    supabase_anon_key: str | None,
    timeout_seconds: int = 5,
) -> str:
    token = extract_bearer_token(authorization)
    if not token:
        raise ValueError("Bearer token required.")
    return fetch_supabase_user_id(
        access_token=token,
        supabase_url=supabase_url,
        supabase_anon_key=supabase_anon_key,
        timeout_seconds=timeout_seconds,
    )
