import unittest

from cortexagent.services.orchestrator import (
    _evaluate_gmail_step_execution,
    _is_gmail_send_confirmation_required_items,
    _is_gmail_sent_items,
)
from cortexagent.tools.base import ToolResultItem


class OrchestratorGmailExecutionHelpersTests(unittest.TestCase):
    def test_is_gmail_sent_items_matches_sent_title_prefix(self):
        items = [ToolResultItem(title="[Sent] Email", url="https://mail.google.com", snippet="ok")]
        self.assertTrue(_is_gmail_sent_items(items))

    def test_is_gmail_send_confirmation_required_items_matches_expected_titles(self):
        send_confirmation = [
            ToolResultItem(
                title="Send confirmation required",
                url="https://mail.google.com",
                snippet="Reply with 'confirm' to send",
            )
        ]
        generic_confirmation = [
            ToolResultItem(title="Confirmation required", url="https://mail.google.com", snippet="")
        ]
        self.assertTrue(_is_gmail_send_confirmation_required_items(send_confirmation))
        self.assertTrue(_is_gmail_send_confirmation_required_items(generic_confirmation))

    def test_is_gmail_send_confirmation_required_items_requires_single_item(self):
        items = [
            ToolResultItem(title="Send confirmation required", url="https://mail.google.com", snippet=""),
            ToolResultItem(title="[Drafted] New email", url="https://mail.google.com", snippet=""),
        ]
        self.assertFalse(_is_gmail_send_confirmation_required_items(items))

    def test_evaluate_gmail_step_execution_handles_confirmation_without_name_errors(self):
        items = [
            ToolResultItem(
                title="Send confirmation required",
                url="https://mail.google.com",
                snippet="Reply with 'confirm' to send",
            )
        ]
        execution_status, reason = _evaluate_gmail_step_execution("gmail_read", items)
        self.assertEqual(execution_status, "action_required")
        self.assertEqual(reason, "gmail_send_confirmation_required")


if __name__ == "__main__":
    unittest.main()
