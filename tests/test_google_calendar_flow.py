import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from cortexagent.router.intent_router import decide_action
from cortexagent.services.connected_accounts_repo import ConnectedAccount, ResolvedProviderToken
from cortexagent.services.google_oauth import GoogleTokenExchange
from cortexagent.services.orchestrator import AgentOrchestrator
from cortexagent.tools.base import Tool, ToolContext, ToolResult, ToolResultItem
from cortexagent.tools.registry import ToolRegistry


@dataclass
class _FakeLtmClient:
    def chat(self, thread_id, text, short_term_limit, authorization):
        return "fallback"

    def add_event(self, thread_id, actor, content, meta, authorization):
        return "event-1"


class _FakeGoogleCalendarTool(Tool):
    name = "google_calendar"

    def run(self, context: ToolContext) -> ToolResult:
        _ = context.tool_meta or {}
        return ToolResult(
            tool_name=self.name,
            query=context.user_text,
            items=[
                ToolResultItem(
                    title="Team Sync",
                    url="https://calendar.google.com/event?id=1",
                    snippet="Starts: 2026-02-15T17:00:00Z",
                )
            ],
        )


class _FakeRepo:
    def __init__(self, resolved: ResolvedProviderToken | None) -> None:
        self._resolved = resolved
        self.upserts = []

    def resolve_provider_token(self, user_id, provider):
        _ = (user_id, provider)
        return self._resolved

    def upsert_active_account(self, payload):
        self.upserts.append(payload)
        return payload


class _FakeGoogleOAuth:
    def __init__(self, refresh_result: GoogleTokenExchange | Exception) -> None:
        self._refresh_result = refresh_result

    def refresh_access_token(self, refresh_token: str) -> GoogleTokenExchange:
        _ = refresh_token
        if isinstance(self._refresh_result, Exception):
            raise self._refresh_result
        return self._refresh_result


def _active_account() -> ConnectedAccount:
    return ConnectedAccount(
        id="acc-1",
        user_id="user-1",
        provider="google",
        provider_account_id="google-sub-1",
        access_token="access-1",
        refresh_token="refresh-1",
        token_type="Bearer",
        scope="openid profile email https://www.googleapis.com/auth/calendar.readonly",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=5),
        status="active",
        meta={},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        deleted_at=None,
    )


class GoogleCalendarFlowTests(unittest.TestCase):
    def test_router_picks_google_calendar(self):
        result = decide_action(
            user_text="What is on my calendar tomorrow?",
            tools_enabled=True,
            web_search_enabled=True,
        )
        self.assertEqual(result.action, "google_calendar")

    def test_orchestrator_refreshes_google_token_when_expired(self):
        account = _active_account()
        resolved = ResolvedProviderToken(
            access_token=None,
            refresh_token=account.refresh_token,
            expires_at=account.expires_at,
            scope=account.scope,
            token_type=account.token_type,
            is_access_token_expired=True,
            account=account,
        )
        repo = _FakeRepo(resolved=resolved)
        oauth = _FakeGoogleOAuth(
            GoogleTokenExchange(
                access_token="new-access",
                refresh_token="new-refresh",
                token_type="Bearer",
                scope=account.scope,
                expires_in=3600,
            )
        )
        registry = ToolRegistry()
        registry.register(_FakeGoogleCalendarTool())

        orchestrator = AgentOrchestrator(
            ltm_client=_FakeLtmClient(),
            tool_registry=registry,
            connected_accounts_repo=repo,  # type: ignore[arg-type]
            google_oauth=oauth,  # type: ignore[arg-type]
        )
        with patch(
            "cortexagent.services.orchestrator.resolve_user_id_from_authorization",
            return_value="user-1",
        ):
            result = orchestrator.handle_chat(
                thread_id="thread-1",
                text="show my upcoming calendar events",
                short_term_limit=30,
                authorization="Bearer token",
            )

        self.assertIn("Upcoming Google Calendar events", result.response)
        self.assertEqual(result.decision.action, "google_calendar")
        self.assertEqual(len(repo.upserts), 1)

    def test_orchestrator_returns_reauth_message_when_refresh_fails(self):
        account = _active_account()
        resolved = ResolvedProviderToken(
            access_token=None,
            refresh_token=account.refresh_token,
            expires_at=account.expires_at,
            scope=account.scope,
            token_type=account.token_type,
            is_access_token_expired=True,
            account=account,
        )
        repo = _FakeRepo(resolved=resolved)
        oauth = _FakeGoogleOAuth(RuntimeError("invalid_grant"))
        registry = ToolRegistry()
        registry.register(_FakeGoogleCalendarTool())
        orchestrator = AgentOrchestrator(
            ltm_client=_FakeLtmClient(),
            tool_registry=registry,
            connected_accounts_repo=repo,  # type: ignore[arg-type]
            google_oauth=oauth,  # type: ignore[arg-type]
        )
        with patch(
            "cortexagent.services.orchestrator.resolve_user_id_from_authorization",
            return_value="user-1",
        ):
            result = orchestrator.handle_chat(
                thread_id="thread-1",
                text="calendar events",
                short_term_limit=30,
                authorization="Bearer token",
            )
        self.assertIn("Please reconnect Google Calendar", result.response)
        self.assertEqual(result.decision.reason, "google_calendar_auth_failed")


if __name__ == "__main__":
    unittest.main()
