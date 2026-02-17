import unittest
from unittest.mock import patch

from cortexagent.services.confirmation_intent import (
    classify_confirmation_intent_deterministic,
    classify_pending_confirmation_intent,
)
from cortexagent.services.llm_json_client import LLMJsonResponse


class ConfirmationIntentTests(unittest.TestCase):
    def test_detects_noisy_affirmations(self):
        confirmations = ["yehhhh", "yeaaaaaa", "yep", "yuppp", "go ahead", "send it", "do it"]
        for sample in confirmations:
            with self.subTest(sample=sample):
                result = classify_confirmation_intent_deterministic(
                    text=sample,
                    pending_calendar=True,
                    pending_gmail=True,
                )
                self.assertEqual(result.intent, "confirm")

    def test_detects_noisy_negations(self):
        negatives = ["nahhh", "nopeee", "cancel", "stop", "never mind", "dont"]
        for sample in negatives:
            with self.subTest(sample=sample):
                result = classify_confirmation_intent_deterministic(
                    text=sample,
                    pending_calendar=True,
                    pending_gmail=True,
                )
                self.assertEqual(result.intent, "cancel")

    def test_ambiguous_replies_do_not_auto_confirm(self):
        ambiguous = ["i guess maybe", "not sure yet", "should i?", "perhaps later"]
        for sample in ambiguous:
            with self.subTest(sample=sample):
                result = classify_confirmation_intent_deterministic(
                    text=sample,
                    pending_calendar=True,
                    pending_gmail=True,
                )
                self.assertEqual(result.intent, "unknown")

    @patch("cortexagent.services.confirmation_intent.call_json_chat_completion")
    def test_llm_assist_is_used_when_deterministic_is_unknown(self, mock_call):
        mock_call.return_value = LLMJsonResponse(
            data={"intent": "confirm", "confidence": 0.99, "reason": "affirmative"},
            error=None,
        )

        result = classify_pending_confirmation_intent(
            text="absolutely",
            pending_calendar=True,
            pending_gmail=False,
        )

        self.assertEqual(result.intent, "confirm")
        self.assertEqual(result.source, "llm")


if __name__ == "__main__":
    unittest.main()
