import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from cortexagent.services.connected_accounts_repo import ConnectedAccount, ResolvedProviderToken
from cortexagent.services.google_oauth import GoogleTokenExchange
from cortexagent.services.orchestrator import AgentOrchestrator
from cortexagent.services.planner import PlannerDecision, PlannerStep
from cortexagent.tools.base import Tool, ToolContext, ToolResult, ToolResultItem
from cortexagent.tools.registry import ToolRegistry


@dataclass
class _FakeLtmClient:
    def chat(self, thread_id, text, short_term_limit, authorization):
        _ = (thread_id, text, short_term_limit, authorization)
        return "fallback"

    def add_event(self, thread_id, actor, content, meta, authorization):
        _ = (thread_id, actor, content, meta, authorization)
        return "event-1"


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
                    title="Drive file",
                    url="https://drive.google.com/file/d/123/view",
                    snippet="Owner: Me",
                )
            ],
        )


class _FakeGoogleCalendarCreateTool(Tool):
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
                    title="[Created] Meeting",
                    url="https://calendar.google.com/calendar/event?eid=test-created",
                    snippet="Created event | Starts: Tue, Feb 18 at 1:00 PM | York",
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
        scope=(
            "openid profile email "
            "https://www.googleapis.com/auth/calendar.events "
            "https://www.googleapis.com/auth/drive.metadata.readonly"
        ),
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=55),
        status="active",
        meta={},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        deleted_at=None,
    )


def _planner_failure() -> PlannerDecision:
    return PlannerDecision(
        plan_id="plan_fail_1",
        steps=[],
        metadata={
            "planner_used": False,
            "planner_attempted": True,
            "planner_source": "fallback",
            "planner_provider": "groq",
            "planner_model": "llama",
            "planner_confidence": 0.0,
            "planner_reason": "llm_request_failed",
            "validation_result": "missing_api_key",
            "fallback_reason": "missing_api_key",
        },
    )


class OrchestratorIntentFallbackRegressionsTests(unittest.TestCase):
    def test_casual_drive_phrase_does_not_route_google_drive(self):
        registry = ToolRegistry()
        registry.register(_FakeGoogleDriveTool())
        orchestrator = AgentOrchestrator(
            ltm_client=_FakeLtmClient(),
            tool_registry=registry,
            connected_accounts_repo=None,  # type: ignore[arg-type]
            google_oauth=None,  # type: ignore[arg-type]
        )

        with patch(
            "cortexagent.services.orchestrator.build_orchestration_plan",
            return_value=_planner_failure(),
        ), patch(
            "cortexagent.services.orchestrator.decide_action",
            return_value=type("Route", (), {"action": "chat", "reason": "offline", "confidence": 0.0})(),
        ):
            result = orchestrator.handle_chat(
                thread_id="thread-1",
                text="i just bought a new car i cant wait to drive",
                short_term_limit=30,
                authorization=None,
            )

        self.assertEqual(result.decision.action, "chat")

    def test_explicit_check_my_drive_routes_google_drive(self):
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
        drive_tool = _FakeGoogleDriveTool()
        registry = ToolRegistry()
        registry.register(drive_tool)
        orchestrator = AgentOrchestrator(
            ltm_client=_FakeLtmClient(),
            tool_registry=registry,
            connected_accounts_repo=repo,  # type: ignore[arg-type]
            google_oauth=oauth,  # type: ignore[arg-type]
        )

        with patch(
            "cortexagent.services.orchestrator.resolve_user_id_from_authorization",
            return_value="user-1",
        ), patch(
            "cortexagent.services.orchestrator.build_orchestration_plan",
            return_value=_planner_failure(),
        ), patch(
            "cortexagent.services.orchestrator.decide_action",
            return_value=type("Route", (), {"action": "chat", "reason": "offline", "confidence": 0.0})(),
        ):
            result = orchestrator.handle_chat(
                thread_id="thread-2",
                text="check my drive",
                short_term_limit=30,
                authorization="Bearer token",
            )

        self.assertEqual(result.decision.action, "orchestration")
        self.assertEqual(len(drive_tool.queries), 1)
        self.assertIn("Google Drive results", result.response)

    def test_calendar_edit_followup_stays_pending_until_confirm(self):
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
        calendar_tool = _FakeGoogleCalendarCreateTool()
        registry = ToolRegistry()
        registry.register(calendar_tool)
        orchestrator = AgentOrchestrator(
            ltm_client=_FakeLtmClient(),
            tool_registry=registry,
            connected_accounts_repo=repo,  # type: ignore[arg-type]
            google_oauth=oauth,  # type: ignore[arg-type]
        )
        planner_pending = PlannerDecision(
            plan_id="plan_pending_1",
            steps=[
                PlannerStep(
                    step_id="step_1",
                    action="google_calendar",
                    operation="create",
                    args={},
                    query="add event appointment on feb 18th at 1:00PM in york",
                    expected_outcome="calendar_write",
                    requires_confirmation=True,
                    depends_on=[],
                    why="calendar_write",
                )
            ],
            metadata={
                "planner_used": True,
                "planner_attempted": True,
                "planner_source": "llm",
                "planner_provider": "groq",
                "planner_model": "llama",
                "planner_confidence": 0.9,
                "planner_reason": "ok",
                "validation_result": "valid",
                "fallback_reason": None,
            },
        )

        with patch(
            "cortexagent.services.orchestrator.resolve_user_id_from_authorization",
            return_value="user-1",
        ), patch(
            "cortexagent.services.orchestrator.build_orchestration_plan",
            return_value=planner_pending,
        ):
            first = orchestrator.handle_chat(
                thread_id="thread-3",
                text="i have an appointment on feb 18th at 1pm in york add it to calendar",
                short_term_limit=30,
                authorization="Bearer token",
            )
            second = orchestrator.handle_chat(
                thread_id="thread-3",
                text="yea but the location should just be york",
                short_term_limit=30,
                authorization="Bearer token",
            )
            third = orchestrator.handle_chat(
                thread_id="thread-3",
                text="confirm",
                short_term_limit=30,
                authorization="Bearer token",
            )

        self.assertIn("Should I add this to Google Calendar?", first.response)
        self.assertIn("still pending confirmation", second.response.lower())
        self.assertEqual(second.decision.reason, "pending_action_edited")
        self.assertEqual(len(calendar_tool.queries), 1)
        self.assertIn(" in York", calendar_tool.queries[0])
        self.assertIn("Google Calendar updated", third.response)


if __name__ == "__main__":
    unittest.main()
