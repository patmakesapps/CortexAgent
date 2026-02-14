from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from .token_security import TokenCipher, build_token_cipher_from_env, redact_sensitive_text


@dataclass(frozen=True)
class ConnectedAccount:
    id: str
    user_id: str
    provider: str
    provider_account_id: str
    access_token: str | None
    refresh_token: str | None
    token_type: str | None
    scope: str | None
    expires_at: datetime | None
    status: str
    meta: dict[str, Any]
    created_at: datetime | None
    updated_at: datetime | None
    deleted_at: datetime | None


@dataclass(frozen=True)
class ResolvedProviderToken:
    access_token: str | None
    refresh_token: str | None
    expires_at: datetime | None
    scope: str | None
    token_type: str | None
    is_access_token_expired: bool
    account: ConnectedAccount


@dataclass(frozen=True)
class ConnectedAccountUpsert:
    user_id: str
    provider: str
    provider_account_id: str
    access_token: str | None = None
    refresh_token: str | None = None
    token_type: str | None = None
    scope: str | None = None
    expires_at: datetime | None = None
    status: str = "active"
    meta: dict[str, Any] | None = None


class ConnectedAccountsRepository:
    def __init__(
        self,
        supabase_url: str | None,
        supabase_service_role_key: str | None,
        table: str = "ltm_connected_accounts",
        timeout_seconds: int = 8,
        token_cipher: TokenCipher | None = None,
    ) -> None:
        self.supabase_url = (supabase_url or "").rstrip("/")
        self.supabase_service_role_key = (supabase_service_role_key or "").strip()
        self.table = (table or "ltm_connected_accounts").strip()
        self.timeout_seconds = max(1, int(timeout_seconds))
        self.token_cipher = token_cipher or build_token_cipher_from_env()

    def is_configured(self) -> bool:
        return bool(self.supabase_url and self.supabase_service_role_key and self.table)

    def get_active_account(
        self,
        user_id: str,
        provider: str,
        provider_account_id: str | None = None,
    ) -> ConnectedAccount | None:
        rows = self._fetch_accounts(
            user_id=user_id,
            provider=provider,
            provider_account_id=provider_account_id,
            only_active=True,
        )
        if not rows:
            return None
        return rows[0]

    def get_active_accounts(
        self,
        user_id: str,
        provider: str,
    ) -> list[ConnectedAccount]:
        return self._fetch_accounts(
            user_id=user_id,
            provider=provider,
            provider_account_id=None,
            only_active=True,
            limit=None,
        )

    def has_active_account(self, user_id: str, provider: str) -> bool:
        return self.get_active_account(user_id=user_id, provider=provider) is not None

    def disconnect_provider(self, user_id: str, provider: str) -> bool:
        accounts = self.get_active_accounts(user_id=user_id, provider=provider)
        if not accounts:
            return False
        for account in accounts:
            self.soft_delete_account(account.id)
        return True

    def upsert_active_account(self, payload: ConnectedAccountUpsert) -> ConnectedAccount:
        existing = self.get_active_account(
            user_id=payload.user_id,
            provider=payload.provider,
            provider_account_id=payload.provider_account_id,
        )
        if existing is None:
            return self._insert_account(payload)
        return self._patch_account(
            account_id=existing.id,
            patch={
                "access_token": payload.access_token,
                "refresh_token": payload.refresh_token,
                "token_type": payload.token_type,
                "scope": payload.scope,
                "expires_at": _to_iso(payload.expires_at),
                "status": payload.status,
                "meta": payload.meta or {},
                "deleted_at": None,
            },
        )

    def revoke_account(self, account_id: str, status: str = "revoked") -> ConnectedAccount:
        if status not in {"revoked", "expired", "error"}:
            raise ValueError("status must be one of: revoked, expired, error")
        return self._patch_account(account_id=account_id, patch={"status": status})

    def soft_delete_account(self, account_id: str) -> ConnectedAccount:
        return self._patch_account(
            account_id=account_id,
            patch={"deleted_at": _to_iso(datetime.now(timezone.utc))},
        )

    def resolve_provider_token(
        self,
        user_id: str,
        provider: str,
        provider_account_id: str | None = None,
        expiry_grace_seconds: int = 30,
    ) -> ResolvedProviderToken | None:
        account = self.get_active_account(
            user_id=user_id,
            provider=provider,
            provider_account_id=provider_account_id,
        )
        if account is None:
            return None

        grace = max(0, expiry_grace_seconds)
        cutoff = datetime.now(timezone.utc) + timedelta(seconds=grace)
        is_expired = account.expires_at is not None and account.expires_at <= cutoff

        return ResolvedProviderToken(
            access_token=None if is_expired else account.access_token,
            refresh_token=account.refresh_token,
            expires_at=account.expires_at,
            scope=account.scope,
            token_type=account.token_type,
            is_access_token_expired=is_expired,
            account=account,
        )

    def _fetch_accounts(
        self,
        user_id: str,
        provider: str,
        provider_account_id: str | None,
        only_active: bool,
        limit: int | None = 1,
    ) -> list[ConnectedAccount]:
        self._ensure_configured()
        url = self._table_url()
        params: dict[str, str] = {
            "select": (
                "id,user_id,provider,provider_account_id,access_token,refresh_token,"
                "token_type,scope,expires_at,status,meta,created_at,updated_at,deleted_at"
            ),
            "user_id": f"eq.{user_id}",
            "provider": f"eq.{provider.strip().lower()}",
            "deleted_at": "is.null",
            "order": "updated_at.desc",
        }
        if isinstance(limit, int) and limit > 0:
            params["limit"] = str(limit)
        if only_active:
            params["status"] = "eq.active"
        if provider_account_id:
            params["provider_account_id"] = f"eq.{provider_account_id}"

        response = requests.get(
            url,
            headers=self._headers(),
            params=params,
            timeout=self.timeout_seconds,
        )
        self._raise_for_error(response, "fetch connected account")
        payload = response.json()
        if not isinstance(payload, list):
            raise RuntimeError("Unexpected connected account response payload.")
        return [
            _to_connected_account(row, token_cipher=self.token_cipher)
            for row in payload
            if isinstance(row, dict)
        ]

    def _insert_account(self, payload: ConnectedAccountUpsert) -> ConnectedAccount:
        self._ensure_configured()
        body = {
            "user_id": payload.user_id,
            "provider": payload.provider.strip().lower(),
            "provider_account_id": payload.provider_account_id,
            "access_token": self._encrypt_token(payload.access_token),
            "refresh_token": self._encrypt_token(payload.refresh_token),
            "token_type": payload.token_type,
            "scope": payload.scope,
            "expires_at": _to_iso(payload.expires_at),
            "status": payload.status,
            "meta": payload.meta or {},
        }
        response = requests.post(
            self._table_url(),
            headers=self._headers(prefer="return=representation"),
            json=body,
            timeout=self.timeout_seconds,
        )
        self._raise_for_error(response, "insert connected account")
        rows = response.json()
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
            raise RuntimeError("Insert connected account returned no rows.")
        return _to_connected_account(rows[0], token_cipher=self.token_cipher)

    def _patch_account(self, account_id: str, patch: dict[str, Any]) -> ConnectedAccount:
        self._ensure_configured()
        clean_patch = {
            key: self._encrypt_token(value)
            if key in {"access_token", "refresh_token"}
            else value
            for key, value in patch.items()
        }
        response = requests.patch(
            self._table_url(),
            headers=self._headers(prefer="return=representation"),
            params={"id": f"eq.{account_id}"},
            json=clean_patch,
            timeout=self.timeout_seconds,
        )
        self._raise_for_error(response, "update connected account")
        rows = response.json()
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
            raise RuntimeError("Update connected account returned no rows.")
        return _to_connected_account(rows[0], token_cipher=self.token_cipher)

    def _headers(self, prefer: str | None = None) -> dict[str, str]:
        headers = {
            "apikey": self.supabase_service_role_key,
            "Authorization": f"Bearer {self.supabase_service_role_key}",
            "Content-Type": "application/json",
        }
        if prefer:
            headers["Prefer"] = prefer
        return headers

    def _table_url(self) -> str:
        return f"{self.supabase_url}/rest/v1/{self.table}"

    def _ensure_configured(self) -> None:
        if self.is_configured():
            return
        raise RuntimeError(
            "Connected accounts repository is not configured. "
            "Set SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, and CONNECTED_ACCOUNTS_TABLE."
        )

    @staticmethod
    def _raise_for_error(response: requests.Response, action: str) -> None:
        if response.ok:
            return
        detail = redact_sensitive_text(response.text.strip())
        raise RuntimeError(
            f"Failed to {action}: HTTP {response.status_code} {detail or 'request failed'}"
        )

    def _encrypt_token(self, value: str | None) -> str | None:
        return self.token_cipher.encrypt(value)


def _to_connected_account(
    row: dict[str, Any], token_cipher: TokenCipher | None = None
) -> ConnectedAccount:
    cipher = token_cipher or build_token_cipher_from_env()
    return ConnectedAccount(
        id=str(row.get("id", "")),
        user_id=str(row.get("user_id", "")),
        provider=str(row.get("provider", "")),
        provider_account_id=str(row.get("provider_account_id", "")),
        access_token=cipher.decrypt(_opt_str(row.get("access_token"))),
        refresh_token=cipher.decrypt(_opt_str(row.get("refresh_token"))),
        token_type=_opt_str(row.get("token_type")),
        scope=_opt_str(row.get("scope")),
        expires_at=_parse_time(row.get("expires_at")),
        status=str(row.get("status", "")),
        meta=row.get("meta") if isinstance(row.get("meta"), dict) else {},
        created_at=_parse_time(row.get("created_at")),
        updated_at=_parse_time(row.get("updated_at")),
        deleted_at=_parse_time(row.get("deleted_at")),
    )


def _opt_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value != "" else None


def _to_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
