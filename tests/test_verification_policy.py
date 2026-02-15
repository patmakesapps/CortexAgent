import unittest

from cortexagent.services.verification import assess_verification_profile


class VerificationPolicyTests(unittest.TestCase):
    def test_shopping_prompt_not_escalated_to_high_risk(self):
        profile = assess_verification_profile("i want to buy a new tractor its for regular lawn care")
        self.assertEqual(profile.level, "medium")
        self.assertTrue(profile.requires_web_verification)
        self.assertEqual(profile.min_independent_sources, 1)

    def test_boat_shopping_prompt_not_escalated_to_high_risk(self):
        profile = assess_verification_profile("need best cruising boats under 200k for 3-5 people")
        self.assertEqual(profile.level, "medium")
        self.assertTrue(profile.requires_web_verification)
        self.assertEqual(profile.min_independent_sources, 1)


if __name__ == "__main__":
    unittest.main()
