import unittest

from cortexagent.services.orchestrator import (
    _build_followup_web_query,
    _extract_link_followup_target,
    _is_link_followup_request,
    _is_link_only_web_followup_request,
)


class WebSearchFollowupTests(unittest.TestCase):
    def test_detects_link_only_followup_variants(self):
        self.assertTrue(_is_link_only_web_followup_request("can you link me"))
        self.assertTrue(_is_link_only_web_followup_request("link me"))
        self.assertTrue(_is_link_only_web_followup_request("send me links"))
        self.assertFalse(_is_link_only_web_followup_request("send me links for center console boats"))

    def test_build_followup_query_reuses_previous_query_for_link_only_prompt(self):
        query = _build_followup_web_query(
            followup_text="can you link me",
            previous_query="best cruising boats under 200k for 3-5 people",
        )
        self.assertEqual(query, "best cruising boats under 200k for 3-5 people")

    def test_detects_link_followup_with_target(self):
        self.assertTrue(_is_link_followup_request("can you link me to teh Dell"))
        self.assertEqual(_extract_link_followup_target("can you link me to teh Dell"), "dell")

    def test_build_followup_query_uses_official_page_for_general_link_request(self):
        query = _build_followup_web_query(
            followup_text="can you link me to OpenAI",
            previous_query="tell me about ai labs",
        )
        self.assertEqual(query, "openai official page")

    def test_build_followup_query_prefers_official_product_page_for_specific_model(self):
        query = _build_followup_web_query(
            followup_text="can you link me to that Dell laptop the Dell XPS 15: 15.6 4K display",
            previous_query="best laptops for gaming and coding",
        )
        self.assertEqual(query, "dell xps 15 official product page buy")

    def test_build_followup_query_prefers_product_page_when_previous_intent_is_purchase(self):
        query = _build_followup_web_query(
            followup_text="can you link me to lenovo legion",
            previous_query="best gaming laptops under 2000 for coding",
        )
        self.assertEqual(query, "lenovo legion official product page buy")

    def test_build_followup_query_maps_few_options_to_contextual_more_options(self):
        query = _build_followup_web_query(
            followup_text="link me to a few options",
            previous_query="i need to buy a new tv 50 inch",
        )
        self.assertEqual(query, "i need to buy a new tv 50 inch more options")


if __name__ == "__main__":
    unittest.main()
