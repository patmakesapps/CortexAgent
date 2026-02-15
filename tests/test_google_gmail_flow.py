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
        _ = (thread_id, text, short_term_limit, authorization)
        return "fallback"

    def add_event(self, thread_id, actor, content, meta, authorization):
        _ = (thread_id, actor, content, meta, authorization)
        return "event-1"


class _FakeConfirmingGoogleGmailTool(Tool):
    name = "google_gmail"

    def run(self, context: ToolContext) -> ToolResult:
        text = context.user_text.strip()
        if text.lower().startswith("confirm send draft"):
            return ToolResult(
                tool_name=self.name,
                query=text,
                items=[
                    ToolResultItem(
                        title="[Sent] Draft r-1",
                        url="https://mail.google.com/mail/u/0/#inbox/thread-1",
                        snippet="Message id: msg-1 | To: me@example.com",
                    )
                ],
            )
        return ToolResult(
            tool_name=self.name,
            query=text,
            items=[
                ToolResultItem(
                    title="Send confirmation required",
                    url="https://mail.google.com/mail/u/0/#drafts?compose=r-1",
                    snippet=(
                        "I am ready to send this draft:\n"
                        "- To: me@example.com\n"
                        "- Subject: Re: Test\n"
                        "- Body: Thanks for your message.\n"
                        "Reply with 'confirm' to send, or 'cancel' to stop."
                    ),
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
        scope=(
            "openid profile email "
            "https://www.googleapis.com/auth/calendar.events "
            "https://www.googleapis.com/auth/gmail.modify"
        ),
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=5),
        status="active",
        meta={},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        deleted_at=None,
    )


class GoogleGmailFlowTests(unittest.TestCase):
    def test_router_picks_web_search_for_live_eth_market_prompt(self):
        result = decide_action(
            user_text="how is ethereum doing on the market right now",
            tools_enabled=True,
            web_search_enabled=True,
        )
        self.assertEqual(result.action, "web_search")

    def test_router_picks_google_gmail(self):
        result = decide_action(
            user_text="List my recent gmail threads",
            tools_enabled=True,
            web_search_enabled=True,
        )
        self.assertEqual(result.action, "google_gmail")

    def test_router_picks_google_gmail_for_check_email_phrase(self):
        result = decide_action(
            user_text="check my email",
            tools_enabled=True,
            web_search_enabled=True,
        )
        self.assertEqual(result.action, "google_gmail")

    def test_router_picks_google_gmail_for_send_email_phrase(self):
        result = decide_action(
            user_text="send an email to anthony@example.com",
            tools_enabled=True,
            web_search_enabled=True,
        )
        self.assertEqual(result.action, "google_gmail")

    def test_router_picks_google_drive(self):
        result = decide_action(
            user_text="find file roadmap in my google drive",
            tools_enabled=True,
            web_search_enabled=True,
        )
        self.assertEqual(result.action, "google_drive")

    def test_router_picks_google_gmail_for_typo_send_email_phrase(self):
        result = decide_action(
            user_text="sned an email to anthony@example.com saying hey",
            tools_enabled=True,
            web_search_enabled=True,
        )
        self.assertEqual(result.action, "google_gmail")

    def test_orchestrator_handles_gmail_send_confirmation_followup(self):
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
        registry.register(_FakeConfirmingGoogleGmailTool())
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
                text="send draft r-1",
                short_term_limit=30,
                authorization="Bearer token",
            )
            second = orchestrator.handle_chat(
                thread_id="thread-1",
                text="confirm",
                short_term_limit=30,
                authorization="Bearer token",
            )

        self.assertIn("Reply with 'confirm' to send", first.response)
        self.assertIn("Gmail sent successfully", second.response)
        self.assertIn("Message id: msg-1", second.response)


if __name__ == "__main__":
    unittest.main()
