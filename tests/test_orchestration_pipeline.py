import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from cortexagent.services.connected_accounts_repo import ConnectedAccount, ResolvedProviderToken
from cortexagent.services.google_oauth import GoogleTokenExchange
from cortexagent.services.orchestrator import AgentOrchestrator
from cortexagent.tools.base import Tool, ToolContext, ToolResult, ToolResultItem
from cortexagent.tools.registry import ToolRegistry


@dataclass
class _FakeLtmClient:
    events: list[dict[str, object]]

    def chat(self, thread_id, text, short_term_limit, authorization):
        _ = (thread_id, text, short_term_limit, authorization)
        return "fallback"

    def add_event(self, thread_id, actor, content, meta, authorization):
        _ = authorization
        self.events.append(
            {
                "thread_id": thread_id,
                "actor": actor,
                "content": content,
                "meta": meta,
            }
        )
        return "event-1"


class _FakeGoogleGmailTool(Tool):
    name = "google_gmail"

    def __init__(self) -> None:
        self.queries: list[str] = []

    def run(self, context: ToolContext) -> ToolResult:
        self.queries.append(context.user_text)
        return ToolResult(
            tool_name=self.name,
            query=context.user_text,
            items=[
                ToolResultItem(
                    title="Any thoughts on Plaid?",
                    url="https://mail.google.com/mail/u/0/#inbox/fake-thread",
                    snippet="From: Anya Hasija <ahasija@plaid.com>",
                )
            ],
        )


class _FakeGoogleDriveTool(Tool):
    name = "google_drive"

    def __init__(self) -> None:
        self.queries: list[str] = []

    def run(self, context: ToolContext) -> ToolResult:
        self.queries.append(context.user_text)
        return ToolResult(
            tool_name=self.name,
            query=context.user_text,
            items=[
                ToolResultItem(
                    title="Plaid interview prep doc",
                    url="https://drive.google.com/file/d/fake-doc/view",
                    snippet="Owner: Me | Updated: Today",
                )
            ],
        )


class _FakeGoogleCalendarTool(Tool):
    name = "google_calendar"

    def __init__(self) -> None:
        self.queries: list[str] = []

    def run(self, context: ToolContext) -> ToolResult:
        self.queries.append(context.user_text)
        return ToolResult(
            tool_name=self.name,
            query=context.user_text,
            items=[
                ToolResultItem(
                    title="[Created] Plaid follow-up",
                    url="https://calendar.google.com/calendar/event?eid=fake-eid-1",
                    snippet="Created event | Starts: Tue, Feb 17 at 3:00 PM",
                )
            ],
        )


class _FakeWebSearchTool(Tool):
    name = "web_search"

    def __init__(self) -> None:
        self.queries: list[str] = []

    def run(self, context: ToolContext) -> ToolResult:
        self.queries.append(context.user_text)
        return ToolResult(
            tool_name=self.name,
            query=context.user_text,
            items=[
                ToolResultItem(
                    title="Official docs",
                    url="https://example.com/docs",
                    snippet="Reference link",
                )
            ],
        )


class _FakeRepo:
    def __init__(self, resolved: ResolvedProviderToken | None) -> None:
        self._resolved = resolved

    def resolve_provider_token(self, user_id, provider):
        _ = (user_id, provider)
        return self._resolved

    def upsert_active_account(self, payload):
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
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=55),
        status="active",
        meta={},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        deleted_at=None,
    )


class OrchestrationPipelineTests(unittest.TestCase):
    def test_orchestrator_executes_multi_step_pipeline_and_persists_structure(self):
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
        gmail_tool = _FakeGoogleGmailTool()
        drive_tool = _FakeGoogleDriveTool()
        calendar_tool = _FakeGoogleCalendarTool()
        registry = ToolRegistry()
        registry.register(gmail_tool)
        registry.register(drive_tool)
        registry.register(calendar_tool)
        ltm_client = _FakeLtmClient(events=[])

        orchestrator = AgentOrchestrator(
            ltm_client=ltm_client,  # type: ignore[arg-type]
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
                text=(
                    "Summarize this email thread, create a meeting, "
                    "attach the relevant Drive doc, and remind me tomorrow."
                ),
                short_term_limit=30,
                authorization="Bearer token",
            )

        self.assertEqual(result.decision.action, "orchestration")
        self.assertIn("Orchestration Summary", result.response)
        self.assertEqual(len(gmail_tool.queries), 1)
        self.assertEqual(len(drive_tool.queries), 1)
        self.assertEqual(len(calendar_tool.queries), 1)
        self.assertGreaterEqual(len(ltm_client.events), 2)
        assistant_events = [event for event in ltm_client.events if event["actor"] == "assistant"]
        self.assertTrue(assistant_events)
        assistant_meta = assistant_events[-1]["meta"]
        self.assertIsInstance(assistant_meta, dict)
        pipeline_meta = assistant_meta.get("tool_pipeline")
        self.assertIsInstance(pipeline_meta, dict)
        self.assertEqual(pipeline_meta.get("step_count"), 3)

    def test_orchestrator_executes_web_gmail_drive_calendar_steps(self):
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
        gmail_tool = _FakeGoogleGmailTool()
        drive_tool = _FakeGoogleDriveTool()
        calendar_tool = _FakeGoogleCalendarTool()
        web_tool = _FakeWebSearchTool()
        registry = ToolRegistry()
        registry.register(gmail_tool)
        registry.register(drive_tool)
        registry.register(calendar_tool)
        registry.register(web_tool)
        ltm_client = _FakeLtmClient(events=[])

        orchestrator = AgentOrchestrator(
            ltm_client=ltm_client,  # type: ignore[arg-type]
            tool_registry=registry,
            connected_accounts_repo=repo,  # type: ignore[arg-type]
            google_oauth=oauth,  # type: ignore[arg-type]
        )
        with patch(
            "cortexagent.services.orchestrator.resolve_user_id_from_authorization",
            return_value="user-1",
        ):
            result = orchestrator.handle_chat(
                thread_id="thread-2",
                text=(
                    "Search the web, summarize this email thread, attach the relevant Drive doc, "
                    "and create a meeting tomorrow."
                ),
                short_term_limit=30,
                authorization="Bearer token",
            )

        self.assertEqual(result.decision.action, "orchestration")
        self.assertEqual(len(web_tool.queries), 1)
        self.assertEqual(len(gmail_tool.queries), 1)
        self.assertEqual(len(drive_tool.queries), 1)
        self.assertEqual(len(calendar_tool.queries), 1)
        assistant_events = [event for event in ltm_client.events if event["actor"] == "assistant"]
        self.assertTrue(assistant_events)
        pipeline_meta = assistant_events[-1]["meta"].get("tool_pipeline")
        self.assertIsInstance(pipeline_meta, dict)
        self.assertEqual(pipeline_meta.get("step_count"), 4)

    def test_orchestrator_orders_calendar_before_then_email_and_builds_confirmation_draft_query(self):
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
        gmail_tool = _FakeGoogleGmailTool()
        calendar_tool = _FakeGoogleCalendarTool()
        registry = ToolRegistry()
        registry.register(gmail_tool)
        registry.register(calendar_tool)
        ltm_client = _FakeLtmClient(events=[])

        orchestrator = AgentOrchestrator(
            ltm_client=ltm_client,  # type: ignore[arg-type]
            tool_registry=registry,
            connected_accounts_repo=repo,  # type: ignore[arg-type]
            google_oauth=oauth,  # type: ignore[arg-type]
        )
        with patch(
            "cortexagent.services.orchestrator.resolve_user_id_from_authorization",
            return_value="user-1",
        ):
            result = orchestrator.handle_chat(
                thread_id="thread-3",
                text=(
                    "add my meeting with james from Purple Park Studios to calendar set it for 2pm on Friday "
                    "then email purpleparkstudios@gmail.com confirming the meeting"
                ),
                short_term_limit=30,
                authorization="Bearer token",
            )

        self.assertEqual(result.decision.action, "orchestration")
        assistant_events = [event for event in ltm_client.events if event["actor"] == "assistant"]
        self.assertTrue(assistant_events)
        pipeline_meta = assistant_events[-1]["meta"].get("tool_pipeline")
        self.assertIsInstance(pipeline_meta, dict)
        steps = pipeline_meta.get("steps")
        self.assertIsInstance(steps, list)
        self.assertGreaterEqual(len(steps), 2)
        self.assertEqual(steps[0].get("action"), "google_calendar")
        self.assertEqual(steps[1].get("action"), "google_gmail")
        self.assertEqual(len(gmail_tool.queries), 1)
        self.assertIn("send an email to purpleparkstudios@gmail.com", gmail_tool.queries[0].lower())
        self.assertIn("confirming our meeting", gmail_tool.queries[0].lower())


if __name__ == "__main__":
    unittest.main()
