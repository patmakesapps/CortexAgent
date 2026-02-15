import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from cortexagent.router.intent_router import decide_action
from cortexagent.services.connected_accounts_repo import ConnectedAccount, ResolvedProviderToken
from cortexagent.services.google_oauth import GoogleTokenExchange
from cortexagent.services.orchestrator import (
    AgentOrchestrator,
    _build_confirmed_calendar_request,
    _looks_like_ad_source,
)
from cortexagent.tools.base import Tool, ToolContext, ToolResult, ToolResultItem
from cortexagent.tools.registry import ToolRegistry


@dataclass
class _FakeLtmClient:
    def chat(self, thread_id, text, short_term_limit, authorization):
        return "fallback"

    def add_event(self, thread_id, actor, content, meta, authorization):
        return "event-1"


@dataclass
class _FailingLtmClient:
    def chat(self, thread_id, text, short_term_limit, authorization):
        return "fallback"

    def add_event(self, thread_id, actor, content, meta, authorization):
        _ = (thread_id, actor, content, meta, authorization)
        raise RuntimeError("persist failed")


@dataclass
class _LyingLtmClient:
    def chat(self, thread_id, text, short_term_limit, authorization):
        _ = (thread_id, text, short_term_limit, authorization)
        return (
            "I've added your meeting to your calendar and set a reminder for Monday at 5:00 PM."
        )

    def add_event(self, thread_id, actor, content, meta, authorization):
        _ = (thread_id, actor, content, meta, authorization)
        return "event-2"


@dataclass
class _LyingEmailLtmClient:
    def chat(self, thread_id, text, short_term_limit, authorization):
        _ = (thread_id, text, short_term_limit, authorization)
        return "I did send the email and it was sent successfully."

    def add_event(self, thread_id, actor, content, meta, authorization):
        _ = (thread_id, actor, content, meta, authorization)
        return "event-3"


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


class _FakeConfirmingGoogleCalendarTool(Tool):
    name = "google_calendar"

    def run(self, context: ToolContext) -> ToolResult:
        text = context.user_text.strip()
        if text.lower().startswith("confirm:"):
            return ToolResult(
                tool_name=self.name,
                query=text,
                items=[
                    ToolResultItem(
                        title="[Created] Meeting with J",
                        url="https://calendar.google.com/calendar/event?eid=test-created-1",
                        snippet="Created event | Starts: Wed, Feb 18 at 11:00 AM | San Francisco",
                    )
                ],
            )
        return ToolResult(
            tool_name=self.name,
            query=text,
            items=[
                ToolResultItem(
                    title="Confirmation required",
                    url="https://calendar.google.com/",
                    snippet=(
                        "I have this draft event:\n- Title: Meeting with Jerry\n"
                        "- Day: Wednesday, Feb 18, 2026\n- Time: 11:00 AM\n"
                        "Should I add this to Google Calendar? "
                        "Reply with 'confirm' to proceed or 'cancel' to stop."
                    ),
                )
            ],
        )


class _CapturingConfirmingGoogleCalendarTool(Tool):
    name = "google_calendar"

    def __init__(self) -> None:
        self.last_user_text = ""

    def run(self, context: ToolContext) -> ToolResult:
        self.last_user_text = context.user_text.strip()
        text = self.last_user_text
        if text.lower().startswith("confirm:"):
            return ToolResult(
                tool_name=self.name,
                query=text,
                items=[
                    ToolResultItem(
                        title="[Created] Meeting with John From Coinbase",
                        url="https://calendar.google.com/calendar/event?eid=test-created-2",
                        snippet="Created event | Starts: Wed, Feb 18 at 11:00 AM | Philly",
                    )
                ],
            )
        return ToolResult(
            tool_name=self.name,
            query=text,
            items=[
                ToolResultItem(
                    title="Confirmation required",
                    url="https://calendar.google.com/",
                    snippet=(
                        "I have this draft event:\n"
                        "- Title: Meeting with John From Coinbase\n"
                        "- Day: Wednesday, Feb 18, 2026\n"
                        "- Time: 11:00 AM\n"
                        "- Location: Philly\n"
                        "Should I add this to Google Calendar? "
                        "Reply with 'confirm' to proceed or 'cancel' to stop."
                    ),
                )
            ],
        )


class _FakeEditableCalendarTool(Tool):
    name = "google_calendar"

    def run(self, context: ToolContext) -> ToolResult:
        text = context.user_text.strip().lower()
        if text.startswith("confirm:"):
            return ToolResult(
                tool_name=self.name,
                query=context.user_text,
                items=[
                    ToolResultItem(
                        title="[Created] Meeting with Jim",
                        url="https://calendar.google.com/calendar/event?eid=created-jim-1",
                        snippet="Created event | Starts: Sat, Feb 14 at 06:00 PM",
                    )
                ],
            )
        if "add event" in text or "add my meeting" in text or "add meeting" in text:
            day = (
                "Saturday, Feb 14, 2026"
                if "today" in text or "saturday" in text
                else "Sunday, Feb 15, 2026"
            )
            return ToolResult(
                tool_name=self.name,
                query=context.user_text,
                items=[
                    ToolResultItem(
                        title="Confirmation required",
                        url="https://calendar.google.com/",
                        snippet=(
                            "I have this draft event:\n"
                            "- Title: Meeting with Jim\n"
                            f"- Day: {day}\n"
                            "- Time: 6:00 PM\n"
                            "Should I add this to Google Calendar? "
                            "Reply with 'confirm' to proceed or 'cancel' to stop."
                        ),
                    )
                ],
            )
        return ToolResult(
            tool_name=self.name,
            query=context.user_text,
            items=[
                ToolResultItem(
                    title="Team Sync",
                    url="https://calendar.google.com/",
                    snippet="Starts: Wed, Feb 18 at 11:00 AM",
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
        scope="openid profile email https://www.googleapis.com/auth/calendar.events",
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

    def test_router_picks_google_calendar_for_create_event(self):
        result = decide_action(
            user_text="Please add event tomorrow at 2pm for project sync.",
            tools_enabled=True,
            web_search_enabled=True,
        )
        self.assertEqual(result.action, "google_calendar")

    def test_router_picks_google_calendar_for_add_my_meeting_phrase(self):
        result = decide_action(
            user_text="add my meeting with Dan and make it for Tuesday at 2:30pm",
            tools_enabled=True,
            web_search_enabled=True,
        )
        self.assertEqual(result.action, "google_calendar")

    def test_router_picks_google_calendar_for_typo_ad_it_to_calendar(self):
        result = decide_action(
            user_text=(
                "i have a meeting at 5pm on monday with danny "
                "please ad it to my calendar"
            ),
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

    def test_orchestrator_routes_short_followup_to_recent_calendar_tool(self):
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
            first = orchestrator.handle_chat(
                thread_id="thread-1",
                text="show my upcoming calendar events",
                short_term_limit=30,
                authorization="Bearer token",
            )
            second = orchestrator.handle_chat(
                thread_id="thread-1",
                text="okay now check",
                short_term_limit=30,
                authorization="Bearer token",
            )

        self.assertEqual(first.decision.action, "google_calendar")
        self.assertEqual(second.decision.action, "google_calendar")
        self.assertEqual(second.decision.reason, "calendar_context_followup")

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

    def test_orchestrator_raises_when_tool_persistence_fails(self):
        account = _active_account()
        resolved = ResolvedProviderToken(
            access_token=account.access_token,
            refresh_token=account.refresh_token,
            expires_at=account.expires_at,
            scope=account.scope,
            token_type=account.token_type,
            is_access_token_expired=False,
            account=account,
        )
        repo = _FakeRepo(resolved=resolved)
        oauth = _FakeGoogleOAuth(
            GoogleTokenExchange(
                access_token="access-1",
                refresh_token="refresh-1",
                token_type="Bearer",
                scope=account.scope,
                expires_in=3600,
            )
        )
        registry = ToolRegistry()
        registry.register(_FakeGoogleCalendarTool())
        orchestrator = AgentOrchestrator(
            ltm_client=_FailingLtmClient(),
            tool_registry=registry,
            connected_accounts_repo=repo,  # type: ignore[arg-type]
            google_oauth=oauth,  # type: ignore[arg-type]
        )
        with patch(
            "cortexagent.services.orchestrator.resolve_user_id_from_authorization",
            return_value="user-1",
        ):
            with self.assertRaises(RuntimeError):
                orchestrator.handle_chat(
                    thread_id="thread-1",
                    text="calendar events",
                    short_term_limit=30,
                    authorization="Bearer token",
                )

    def test_orchestrator_handles_calendar_confirmation_followup_with_edits(self):
        account = _active_account()
        resolved = ResolvedProviderToken(
            access_token=account.access_token,
            refresh_token=account.refresh_token,
            expires_at=account.expires_at,
            scope=account.scope,
            token_type=account.token_type,
            is_access_token_expired=False,
            account=account,
        )
        repo = _FakeRepo(resolved=resolved)
        oauth = _FakeGoogleOAuth(
            GoogleTokenExchange(
                access_token="access-1",
                refresh_token="refresh-1",
                token_type="Bearer",
                scope=account.scope,
                expires_in=3600,
            )
        )
        registry = ToolRegistry()
        registry.register(_FakeConfirmingGoogleCalendarTool())
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
            first = orchestrator.handle_chat(
                thread_id="thread-1",
                text="add my meeting with jerry on wednesday at 11am in san francisco",
                short_term_limit=30,
                authorization="Bearer token",
            )
            second = orchestrator.handle_chat(
                thread_id="thread-1",
                text="yes but change the title to Meeting with J",
                short_term_limit=30,
                authorization="Bearer token",
            )

        self.assertIn("Should I add this to Google Calendar?", first.response)
        self.assertIn("Google Calendar updated", second.response)
        self.assertIn("Link: https://calendar.google.com/calendar/event?eid=test-created-1", second.response)

    def test_orchestrator_accepts_plain_confirm_for_calendar_draft(self):
        account = _active_account()
        resolved = ResolvedProviderToken(
            access_token=account.access_token,
            refresh_token=account.refresh_token,
            expires_at=account.expires_at,
            scope=account.scope,
            token_type=account.token_type,
            is_access_token_expired=False,
            account=account,
        )
        repo = _FakeRepo(resolved=resolved)
        oauth = _FakeGoogleOAuth(
            GoogleTokenExchange(
                access_token="access-1",
                refresh_token="refresh-1",
                token_type="Bearer",
                scope=account.scope,
                expires_in=3600,
            )
        )
        registry = ToolRegistry()
        registry.register(_FakeConfirmingGoogleCalendarTool())
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
            first = orchestrator.handle_chat(
                thread_id="thread-1",
                text="add my meeting with jerry on wednesday at 11am in san francisco",
                short_term_limit=30,
                authorization="Bearer token",
            )
            second = orchestrator.handle_chat(
                thread_id="thread-1",
                text="confirm",
                short_term_limit=30,
                authorization="Bearer token",
            )

        self.assertIn("Should I add this to Google Calendar?", first.response)
        self.assertIn("Google Calendar updated", second.response)
        self.assertIn(
            "Link: https://calendar.google.com/calendar/event?eid=test-created-1",
            second.response,
        )

    def test_orchestrator_accepts_confirm_with_trailing_period_for_calendar_draft(self):
        account = _active_account()
        resolved = ResolvedProviderToken(
            access_token=account.access_token,
            refresh_token=account.refresh_token,
            expires_at=account.expires_at,
            scope=account.scope,
            token_type=account.token_type,
            is_access_token_expired=False,
            account=account,
        )
        repo = _FakeRepo(resolved=resolved)
        oauth = _FakeGoogleOAuth(
            GoogleTokenExchange(
                access_token="access-1",
                refresh_token="refresh-1",
                token_type="Bearer",
                scope=account.scope,
                expires_in=3600,
            )
        )
        registry = ToolRegistry()
        registry.register(_FakeConfirmingGoogleCalendarTool())
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
            first = orchestrator.handle_chat(
                thread_id="thread-1",
                text="add my meeting with jerry on wednesday at 11am in san francisco",
                short_term_limit=30,
                authorization="Bearer token",
            )
            second = orchestrator.handle_chat(
                thread_id="thread-1",
                text="confirm.",
                short_term_limit=30,
                authorization="Bearer token",
            )

        self.assertIn("Should I add this to Google Calendar?", first.response)
        self.assertIn("Google Calendar updated", second.response)
        self.assertIn(
            "Link: https://calendar.google.com/calendar/event?eid=test-created-1",
            second.response,
        )

    def test_orchestrator_accepts_sure_for_calendar_draft(self):
        account = _active_account()
        resolved = ResolvedProviderToken(
            access_token=account.access_token,
            refresh_token=account.refresh_token,
            expires_at=account.expires_at,
            scope=account.scope,
            token_type=account.token_type,
            is_access_token_expired=False,
            account=account,
        )
        repo = _FakeRepo(resolved=resolved)
        oauth = _FakeGoogleOAuth(
            GoogleTokenExchange(
                access_token="access-1",
                refresh_token="refresh-1",
                token_type="Bearer",
                scope=account.scope,
                expires_in=3600,
            )
        )
        registry = ToolRegistry()
        registry.register(_FakeConfirmingGoogleCalendarTool())
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
            first = orchestrator.handle_chat(
                thread_id="thread-1",
                text="add my meeting with jerry on wednesday at 11am in san francisco",
                short_term_limit=30,
                authorization="Bearer token",
            )
            second = orchestrator.handle_chat(
                thread_id="thread-1",
                text="sure",
                short_term_limit=30,
                authorization="Bearer token",
            )

        self.assertIn("Should I add this to Google Calendar?", first.response)
        self.assertIn("Google Calendar updated", second.response)
        self.assertIn(
            "Link: https://calendar.google.com/calendar/event?eid=test-created-1",
            second.response,
        )

    def test_orchestrator_accepts_yea_for_calendar_draft(self):
        account = _active_account()
        resolved = ResolvedProviderToken(
            access_token=account.access_token,
            refresh_token=account.refresh_token,
            expires_at=account.expires_at,
            scope=account.scope,
            token_type=account.token_type,
            is_access_token_expired=False,
            account=account,
        )
        repo = _FakeRepo(resolved=resolved)
        oauth = _FakeGoogleOAuth(
            GoogleTokenExchange(
                access_token="access-1",
                refresh_token="refresh-1",
                token_type="Bearer",
                scope=account.scope,
                expires_in=3600,
            )
        )
        registry = ToolRegistry()
        registry.register(_FakeConfirmingGoogleCalendarTool())
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
            first = orchestrator.handle_chat(
                thread_id="thread-1",
                text="add my meeting with jerry on wednesday at 11am in san francisco",
                short_term_limit=30,
                authorization="Bearer token",
            )
            second = orchestrator.handle_chat(
                thread_id="thread-1",
                text="yea",
                short_term_limit=30,
                authorization="Bearer token",
            )

        self.assertIn("Should I add this to Google Calendar?", first.response)
        self.assertIn("Google Calendar updated", second.response)
        self.assertIn(
            "Link: https://calendar.google.com/calendar/event?eid=test-created-1",
            second.response,
        )

    def test_orchestrator_treats_add_it_followup_as_confirmation(self):
        account = _active_account()
        resolved = ResolvedProviderToken(
            access_token=account.access_token,
            refresh_token=account.refresh_token,
            expires_at=account.expires_at,
            scope=account.scope,
            token_type=account.token_type,
            is_access_token_expired=False,
            account=account,
        )
        repo = _FakeRepo(resolved=resolved)
        oauth = _FakeGoogleOAuth(
            GoogleTokenExchange(
                access_token="access-1",
                refresh_token="refresh-1",
                token_type="Bearer",
                scope=account.scope,
                expires_in=3600,
            )
        )
        registry = ToolRegistry()
        registry.register(_FakeConfirmingGoogleCalendarTool())
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
            first = orchestrator.handle_chat(
                thread_id="thread-1",
                text="add my meeting with jerry on wednesday at 11am in san francisco",
                short_term_limit=30,
                authorization="Bearer token",
            )
            second = orchestrator.handle_chat(
                thread_id="thread-1",
                text="right but i asked you to add my meeting with john",
                short_term_limit=30,
                authorization="Bearer token",
            )

        self.assertIn("Should I add this to Google Calendar?", first.response)
        self.assertIn("Google Calendar updated", second.response)
        self.assertIn("Link: https://calendar.google.com/calendar/event?eid=test-created-1", second.response)

    def test_build_confirmed_request_expands_plain_confirm_with_explicit_fields(self):
        draft = (
            "I have this draft event:\n"
            "- Title: Meeting with John From Coinbase\n"
            "- Day: Wednesday, Feb 18, 2026\n"
            "- Time: 11:00 AM\n"
            "- Location: Philly\n"
            "Should I add this to Google Calendar? "
            "Reply with 'confirm' to proceed or 'cancel' to stop."
        )
        merged = _build_confirmed_calendar_request(draft_text=draft, followup_text="confirm")
        self.assertTrue(merged.lower().startswith("confirm: add event "))
        self.assertIn("Meeting with John From Coinbase", merged)
        self.assertIn("on Wednesday, Feb 18, 2026", merged)
        self.assertIn("at 11:00AM", merged)
        self.assertIn("in Philly", merged)

    def test_orchestrator_blocks_calendar_write_claims_without_calendar_tool(self):
        registry = ToolRegistry()
        registry.register(_FakeGoogleCalendarTool())
        orchestrator = AgentOrchestrator(
            ltm_client=_LyingLtmClient(),
            tool_registry=registry,
            connected_accounts_repo=None,  # type: ignore[arg-type]
            google_oauth=None,  # type: ignore[arg-type]
        )
        result = orchestrator.handle_chat(
            thread_id="thread-1",
            text="that is not what i asked",
            short_term_limit=30,
            authorization=None,
        )
        self.assertNotIn("I've added your meeting to your calendar", result.response)
        self.assertIn("I haven’t added anything to your Google Calendar yet.", result.response)

    def test_orchestrator_blocks_gmail_send_claims_without_gmail_tool(self):
        registry = ToolRegistry()
        registry.register(_FakeGoogleCalendarTool())
        orchestrator = AgentOrchestrator(
            ltm_client=_LyingEmailLtmClient(),
            tool_registry=registry,
            connected_accounts_repo=None,  # type: ignore[arg-type]
            google_oauth=None,  # type: ignore[arg-type]
        )
        result = orchestrator.handle_chat(
            thread_id="thread-1",
            text="did you actually send it?",
            short_term_limit=30,
            authorization=None,
        )
        self.assertNotIn("I did send the email", result.response)
        self.assertIn("I haven’t sent an email yet.", result.response)

    def test_source_filter_rejects_generic_search_homepages(self):
        self.assertTrue(_looks_like_ad_source("Google: Search the world's information", "https://www.google.com/"))
        self.assertTrue(_looks_like_ad_source("Yandex - fast Internet search", "https://yandex.com/"))

    def test_orchestrator_handles_calendar_day_correction_before_ok(self):
        account = _active_account()
        resolved = ResolvedProviderToken(
            access_token=account.access_token,
            refresh_token=account.refresh_token,
            expires_at=account.expires_at,
            scope=account.scope,
            token_type=account.token_type,
            is_access_token_expired=False,
            account=account,
        )
        repo = _FakeRepo(resolved=resolved)
        oauth = _FakeGoogleOAuth(
            GoogleTokenExchange(
                access_token="access-1",
                refresh_token="refresh-1",
                token_type="Bearer",
                scope=account.scope,
                expires_in=3600,
            )
        )
        registry = ToolRegistry()
        registry.register(_FakeEditableCalendarTool())
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
            first = orchestrator.handle_chat(
                thread_id="thread-1",
                text="add meeting with Jim tonight at 6pm to my calendar",
                short_term_limit=30,
                authorization="Bearer token",
            )
            second = orchestrator.handle_chat(
                thread_id="thread-1",
                text="today is saturday tho not sunday",
                short_term_limit=30,
                authorization="Bearer token",
            )
            third = orchestrator.handle_chat(
                thread_id="thread-1",
                text="ok",
                short_term_limit=30,
                authorization="Bearer token",
            )

        self.assertIn("Sunday, Feb 15, 2026", first.response)
        self.assertIn("Saturday, Feb 14, 2026", second.response)
        self.assertIn("Google Calendar updated", third.response)
        self.assertIn(
            "Link: https://calendar.google.com/calendar/event?eid=created-jim-1",
            third.response,
        )


if __name__ == "__main__":
    unittest.main()
