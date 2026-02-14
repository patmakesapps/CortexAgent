import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from cortexagent.services.connected_accounts_repo import ConnectedAccount
from cortexagent.services.google_oauth import GoogleOAuthService, GoogleTokenExchange


class _FakeRepo:
    def __init__(self, existing: ConnectedAccount | None = None) -> None:
        self._existing = existing
        self.last_upsert = None

    def get_active_account(self, user_id, provider, provider_account_id):
        _ = (user_id, provider, provider_account_id)
        return self._existing

    def upsert_active_account(self, payload):
        self.last_upsert = payload
        return ConnectedAccount(
            id="acc-1",
            user_id=payload.user_id,
            provider=payload.provider,
            provider_account_id=payload.provider_account_id,
            access_token=payload.access_token,
            refresh_token=payload.refresh_token,
            token_type=payload.token_type,
            scope=payload.scope,
            expires_at=payload.expires_at,
            status=payload.status,
            meta=payload.meta or {},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            deleted_at=None,
        )


def _existing_account(scope: str | None) -> ConnectedAccount:
    return ConnectedAccount(
        id="acc-existing",
        user_id="user-1",
        provider="google",
        provider_account_id="google-sub-1",
        access_token="old-access",
        refresh_token="old-refresh",
        token_type="Bearer",
        scope=scope,
        expires_at=datetime.now(timezone.utc),
        status="active",
        meta={},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        deleted_at=None,
    )


class GoogleOAuthServiceTests(unittest.TestCase):
    def test_connect_account_backfills_scope_from_tokeninfo_when_missing(self):
        service = GoogleOAuthService(
            client_id="cid",
            client_secret="secret",
            redirect_uri="https://app.example.com/callback",
        )
        repo = _FakeRepo()
        fallback_scope = (
            "openid email profile "
            "https://www.googleapis.com/auth/gmail.readonly "
            "https://www.googleapis.com/auth/gmail.compose"
        )
        with patch.object(
            service,
            "exchange_code",
            return_value=GoogleTokenExchange(
                access_token="new-access",
                refresh_token="new-refresh",
                token_type="Bearer",
                scope=None,
                expires_in=3600,
            ),
        ), patch.object(
            service,
            "fetch_user_info",
            return_value={"sub": "google-sub-1"},
        ), patch.object(
            service,
            "fetch_token_scope",
            return_value=fallback_scope,
        ):
            account = service.connect_account(
                repo=repo, user_id="user-1", code="code-1", code_verifier="verifier-1"
            )

        self.assertIsNotNone(repo.last_upsert)
        merged = repo.last_upsert.scope or ""
        self.assertIn("openid", merged)
        self.assertIn("email", merged)
        self.assertIn("profile", merged)
        self.assertIn("https://www.googleapis.com/auth/gmail.readonly", merged)
        self.assertIn("https://www.googleapis.com/auth/gmail.compose", merged)
        self.assertEqual(account.scope, repo.last_upsert.scope)

    def test_connect_account_keeps_existing_scope_when_new_scope_missing(self):
        service = GoogleOAuthService(
            client_id="cid",
            client_secret="secret",
            redirect_uri="https://app.example.com/callback",
        )
        existing_scope = (
            "openid email profile https://www.googleapis.com/auth/calendar.events"
        )
        repo = _FakeRepo(existing=_existing_account(existing_scope))
        with patch.object(
            service,
            "exchange_code",
            return_value=GoogleTokenExchange(
                access_token="new-access",
                refresh_token=None,
                token_type="Bearer",
                scope=None,
                expires_in=3600,
            ),
        ), patch.object(
            service,
            "fetch_user_info",
            return_value={"sub": "google-sub-1"},
        ), patch.object(service, "fetch_token_scope", return_value=None):
            account = service.connect_account(
                repo=repo, user_id="user-1", code="code-1", code_verifier="verifier-1"
            )

        self.assertIsNotNone(repo.last_upsert)
        merged = repo.last_upsert.scope or ""
        self.assertIn("openid", merged)
        self.assertIn("email", merged)
        self.assertIn("profile", merged)
        self.assertIn("https://www.googleapis.com/auth/calendar.events", merged)
        self.assertEqual(account.scope, repo.last_upsert.scope)

    def test_connect_account_prefers_tokeninfo_scope_over_existing_scope(self):
        service = GoogleOAuthService(
            client_id="cid",
            client_secret="secret",
            redirect_uri="https://app.example.com/callback",
        )
        existing_scope = "openid email profile"
        repo = _FakeRepo(existing=_existing_account(existing_scope))
        tokeninfo_scope = (
            "https://www.googleapis.com/auth/gmail.readonly "
            "https://www.googleapis.com/auth/gmail.compose"
        )
        with patch.object(
            service,
            "exchange_code",
            return_value=GoogleTokenExchange(
                access_token="new-access",
                refresh_token=None,
                token_type="Bearer",
                scope=None,
                expires_in=3600,
            ),
        ), patch.object(
            service,
            "fetch_user_info",
            return_value={"sub": "google-sub-1"},
        ), patch.object(
            service,
            "fetch_token_scope",
            return_value=tokeninfo_scope,
        ):
            account = service.connect_account(
                repo=repo, user_id="user-1", code="code-1", code_verifier="verifier-1"
            )

        self.assertIsNotNone(repo.last_upsert)
        self.assertEqual(repo.last_upsert.scope, tokeninfo_scope)
        self.assertEqual(account.scope, tokeninfo_scope)


if __name__ == "__main__":
    unittest.main()
