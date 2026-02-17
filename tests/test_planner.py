import unittest
from unittest.mock import patch

from cortexagent.services.llm_json_client import LLMJsonResponse
from cortexagent.services.planner import build_orchestration_plan


class PlannerTests(unittest.TestCase):
    @patch("cortexagent.services.planner.call_json_chat_completion")
    def test_planner_returns_valid_steps_when_schema_is_valid(self, mock_call):
        mock_call.return_value = LLMJsonResponse(
            data={
                "steps": [
                    {
                        "step_id": "step_1",
                        "action": "google_gmail",
                        "query": "send an email to test@example.com saying hello",
                        "expected_outcome": "gmail_send",
                        "requires_confirmation": True,
                        "depends_on": [],
                    },
                    {
                        "step_id": "step_2",
                        "action": "google_calendar",
                        "query": "add event follow up tomorrow at 3pm",
                        "expected_outcome": "calendar_write",
                        "requires_confirmation": True,
                        "depends_on": ["step_1"],
                    },
                ],
                "planner_confidence": 0.93,
                "reason": "multi_intent_write_flow",
            },
            error=None,
        )

        decision = build_orchestration_plan(
            user_text="email test@example.com then add it to my calendar tomorrow at 3pm",
            prior_user_text=None,
            prior_tool_action=None,
        )

        self.assertEqual(len(decision.steps), 2)
        self.assertEqual(decision.steps[0].action, "google_gmail")
        self.assertEqual(decision.steps[1].action, "google_calendar")
        self.assertTrue(bool(decision.metadata.get("planner_used")))
        self.assertEqual(decision.metadata.get("validation_result"), "valid")

    @patch("cortexagent.services.planner.call_json_chat_completion")
    def test_planner_rejects_invalid_schema_actions(self, mock_call):
        mock_call.return_value = LLMJsonResponse(
            data={
                "steps": [
                    {
                        "step_id": "step_1",
                        "action": "shell_exec",
                        "query": "do something",
                        "expected_outcome": "chat",
                        "requires_confirmation": False,
                        "depends_on": [],
                    }
                ],
                "planner_confidence": 0.9,
                "reason": "bad_action",
            },
            error=None,
        )

        decision = build_orchestration_plan(
            user_text="do something",
            prior_user_text=None,
            prior_tool_action=None,
        )

        self.assertEqual(decision.steps, [])
        self.assertFalse(bool(decision.metadata.get("planner_used")))
        self.assertEqual(decision.metadata.get("validation_result"), "invalid_steps")
        self.assertEqual(decision.metadata.get("fallback_reason"), "invalid_steps")

    @patch("cortexagent.services.planner.call_json_chat_completion")
    def test_planner_falls_back_on_low_confidence(self, mock_call):
        mock_call.return_value = LLMJsonResponse(
            data={
                "steps": [
                    {
                        "step_id": "step_1",
                        "action": "google_gmail",
                        "query": "check inbox",
                        "expected_outcome": "gmail_read",
                        "requires_confirmation": False,
                        "depends_on": [],
                    },
                    {
                        "step_id": "step_2",
                        "action": "google_drive",
                        "query": "find relevant files",
                        "expected_outcome": "drive_search",
                        "requires_confirmation": False,
                        "depends_on": [],
                    },
                ],
                "planner_confidence": 0.1,
                "reason": "uncertain",
            },
            error=None,
        )

        decision = build_orchestration_plan(
            user_text="check inbox and find relevant files",
            prior_user_text=None,
            prior_tool_action=None,
        )

        self.assertEqual(decision.steps, [])
        self.assertFalse(bool(decision.metadata.get("planner_used")))
        self.assertEqual(decision.metadata.get("validation_result"), "low_confidence")
        self.assertEqual(decision.metadata.get("fallback_reason"), "low_confidence")


if __name__ == "__main__":
    unittest.main()
