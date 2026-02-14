from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import parse_qs

import requests

from .connected_accounts_repo import (
    ConnectedAccount,
    ConnectedAccountsRepository,
    ConnectedAccountUpsert,
)
from .token_security import redact_sensitive_text


@dataclass(frozen=True)
class GoogleTokenExchange:
    access_token: str
    refresh_token: str | None
    token_type: str | None
    scope: str | None
    expires_in: int | None


class GoogleOAuthService:
    TOKEN_URL = "https://oauth2.googleapis.com/token"
    USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"

    def __init__(
        self,
        client_id: str | None,
        client_secret: str | None,
        redirect_uri: str | None,
        timeout_seconds: int = 8,
    ) -> None:
        self.client_id = (client_id or "").strip()
        self.client_secret = (client_secret or "").strip()
        self.redirect_uri = (redirect_uri or "").strip()
        self.timeout_seconds = max(1, timeout_seconds)

    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret and self.redirect_uri)

    def connect_account(
        self,
        repo: ConnectedAccountsRepository,
        user_id: str,
        code: str,
        code_verifier: str | None = None,
    ) -> ConnectedAccount:
        if not self.is_configured():
            raise RuntimeError(
                "Google OAuth is not configured. Set GOOGLE_CLIENT_ID, "
                "GOOGLE_CLIENT_SECRET, and GOOGLE_REDIRECT_URI."
            )
        token = self.exchange_code(code=code, code_verifier=code_verifier)
        user_info = self.fetch_user_info(token.access_token)
        provider_account_id = self._read_provider_account_id(user_info)
        existing = repo.get_active_account(
            user_id=user_id,
            provider="google",
            provider_account_id=provider_account_id,
        )

        expires_at = None
        if isinstance(token.expires_in, int) and token.expires_in > 0:
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=token.expires_in)

        account = repo.upsert_active_account(
            ConnectedAccountUpsert(
                user_id=user_id,
                provider="google",
                provider_account_id=provider_account_id,
                access_token=token.access_token,
                refresh_token=token.refresh_token or (existing.refresh_token if existing else None),
                token_type=token.token_type,
                scope=token.scope,
                expires_at=expires_at,
                status="active",
                meta={
                    "email": user_info.get("email"),
                    "email_verified": user_info.get("email_verified"),
                    "name": user_info.get("name"),
                    "picture": user_info.get("picture"),
                },
            )
        )
        return account

    def refresh_access_token(self, refresh_token: str) -> GoogleTokenExchange:
        if not self.is_configured():
            raise RuntimeError(
                "Google OAuth is not configured. Set GOOGLE_CLIENT_ID, "
                "GOOGLE_CLIENT_SECRET, and GOOGLE_REDIRECT_URI."
            )
        body = {
            "refresh_token": refresh_token.strip(),
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "refresh_token",
        }
        response = requests.post(
            self.TOKEN_URL,
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=self.timeout_seconds,
        )
        if not response.ok:
            detail = _extract_google_error(response)
            raise RuntimeError(f"Google refresh failed: {detail}")

        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Google refresh returned unexpected payload.")
        access_token = payload.get("access_token")
        if not isinstance(access_token, str) or not access_token.strip():
            raise RuntimeError("Google refresh missing access_token.")

        expires_in_raw = payload.get("expires_in")
        expires_in: int | None = None
        if isinstance(expires_in_raw, int):
            expires_in = expires_in_raw
        elif isinstance(expires_in_raw, str) and expires_in_raw.isdigit():
            expires_in = int(expires_in_raw)

        return GoogleTokenExchange(
            access_token=access_token.strip(),
            refresh_token=_opt_str(payload.get("refresh_token")),
            token_type=_opt_str(payload.get("token_type")),
            scope=_opt_str(payload.get("scope")),
            expires_in=expires_in,
        )

    def exchange_code(
        self, code: str, code_verifier: str | None = None
    ) -> GoogleTokenExchange:
        body = {
            "code": code.strip(),
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "redirect_uri": self.redirect_uri,
            "grant_type": "authorization_code",
        }
        if code_verifier:
            body["code_verifier"] = code_verifier.strip()

        response = requests.post(
            self.TOKEN_URL,
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=self.timeout_seconds,
        )
        if not response.ok:
            detail = _extract_google_error(response)
            raise RuntimeError(f"Google token exchange failed: {detail}")

        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Google token exchange returned unexpected payload.")
        access_token = payload.get("access_token")
        if not isinstance(access_token, str) or not access_token.strip():
            raise RuntimeError("Google token exchange missing access_token.")

        expires_in_raw = payload.get("expires_in")
        expires_in: int | None = None
        if isinstance(expires_in_raw, int):
            expires_in = expires_in_raw
        elif isinstance(expires_in_raw, str) and expires_in_raw.isdigit():
            expires_in = int(expires_in_raw)

        return GoogleTokenExchange(
            access_token=access_token.strip(),
            refresh_token=_opt_str(payload.get("refresh_token")),
            token_type=_opt_str(payload.get("token_type")),
            scope=_opt_str(payload.get("scope")),
            expires_in=expires_in,
        )

    def fetch_user_info(self, access_token: str) -> dict[str, Any]:
        response = requests.get(
            self.USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=self.timeout_seconds,
        )
        if not response.ok:
            raise RuntimeError(
                "Google userinfo fetch failed: "
                f"HTTP {response.status_code} {redact_sensitive_text(response.text.strip())}"
            )
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Google userinfo returned unexpected payload.")
        return payload

    @staticmethod
    def _read_provider_account_id(user_info: dict[str, Any]) -> str:
        sub = user_info.get("sub")
        if not isinstance(sub, str) or not sub.strip():
            raise RuntimeError(
                "Google userinfo missing 'sub'. Ensure 'openid' scope is included."
            )
        return sub.strip()


def _opt_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _extract_google_error(response: requests.Response) -> str:
    text = redact_sensitive_text(response.text.strip())
    try:
        payload = response.json()
    except Exception:
        return text or f"HTTP {response.status_code}"
    if isinstance(payload, dict):
        err = payload.get("error")
        desc = payload.get("error_description")
        if isinstance(err, str) and isinstance(desc, str):
            return f"{err}: {desc}"
        if isinstance(err, str):
            return err
    if not text:
        return f"HTTP {response.status_code}"
    # Handle x-www-form-urlencoded style errors defensively.
    if "=" in text and "&" in text:
        parsed = parse_qs(text, keep_blank_values=True)
        err = parsed.get("error", [""])[0]
        desc = parsed.get("error_description", [""])[0]
        if err and desc:
            return f"{err}: {desc}"
    return text
