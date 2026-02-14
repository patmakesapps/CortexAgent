import unittest
from unittest.mock import patch

from cortexagent.tools.base import ToolContext, ToolResultItem
from cortexagent.tools.google_calendar import GoogleCalendarTool


class GoogleCalendarToolWriteConfirmationTests(unittest.TestCase):
    def test_create_request_requires_confirmation(self):
        tool = GoogleCalendarTool()
        result = tool.run(
            ToolContext(
                thread_id="thread-1",
                user_text="add event tomorrow at 2pm project sync",
                tool_meta={"access_token": "token-1"},
            )
        )
        self.assertEqual(result.items[0].title, "Confirmation required")
        self.assertIn("reply with 'confirm'", result.items[0].snippet.lower())
        self.assertIn("- Title:", result.items[0].snippet)

    def test_add_my_meeting_phrase_requires_confirmation(self):
        tool = GoogleCalendarTool()
        result = tool.run(
            ToolContext(
                thread_id="thread-1",
                user_text="add my meeting with dan bakeman on february 15th at 2:30pm",
                tool_meta={"access_token": "token-1"},
            )
        )
        self.assertEqual(result.items[0].title, "Confirmation required")
        self.assertIn("Should I add this to Google Calendar?", result.items[0].snippet)
        self.assertIn("Meeting with Dan Bakeman", result.items[0].snippet)

    def test_missing_date_and_time_adds_future_day_assumption(self):
        tool = GoogleCalendarTool()
        result = tool.run(
            ToolContext(
                thread_id="thread-1",
                user_text="add meeting with dan",
                tool_meta={"access_token": "token-1"},
            )
        )
        self.assertEqual(result.items[0].title, "Confirmation required")
        self.assertIn("closest upcoming day in the future", result.items[0].snippet)
        self.assertIn("I assumed the time: 09:00 AM.", result.items[0].snippet)
        self.assertIn("Assumptions to confirm:", result.items[0].snippet)

    def test_title_extraction_stops_before_trailing_instruction_words(self):
        tool = GoogleCalendarTool()
        result = tool.run(
            ToolContext(
                thread_id="thread-1",
                user_text=(
                    "add my meeting with the dev team in paris and make the meeting time 5pm "
                    "on thursday"
                ),
                tool_meta={"access_token": "token-1"},
            )
        )
        self.assertIn("- Title: Meeting with The Dev Team", result.items[0].snippet)

    def test_detects_add_it_to_google_calendar_phrasing(self):
        tool = GoogleCalendarTool()
        result = tool.run(
            ToolContext(
                thread_id="thread-1",
                user_text=(
                    "i have a meeting on wednesday with john from coinbase "
                    "its for 4pm please add it to my google calendar"
                ),
                tool_meta={"access_token": "token-1"},
            )
        )
        self.assertEqual(result.items[0].title, "Confirmation required")
        self.assertIn("- Time: 4:00 PM", result.items[0].snippet)
        self.assertIn("- Title: Meeting with John From Coinbase", result.items[0].snippet)
        self.assertNotIn("<event details>", result.items[0].snippet)

    def test_detects_typo_ad_it_to_calendar_phrasing(self):
        tool = GoogleCalendarTool()
        result = tool.run(
            ToolContext(
                thread_id="thread-1",
                user_text=(
                    "i have a meeting at 5pm on monday with danny "
                    "its at 11 south york street in philadelphia please ad it to my calendar"
                ),
                tool_meta={"access_token": "token-1"},
            )
        )
        self.assertEqual(result.items[0].title, "Confirmation required")
        self.assertIn("Should I add this to Google Calendar?", result.items[0].snippet)

    def test_confirmed_create_request_executes_write(self):
        tool = GoogleCalendarTool()
        with patch.object(
            tool,
            "_quick_add_event",
            return_value=ToolResultItem(
                title="[Created] Project Sync",
                url="https://calendar.google.com/",
                snippet="Created event | Starts: Fri, Feb 14 at 02:00 PM",
            ),
        ) as quick_add_mock, patch.object(
            tool,
            "_list_upcoming_events",
            return_value=[],
        ) as list_mock:
            result = tool.run(
                ToolContext(
                    thread_id="thread-1",
                    user_text="confirm: add event tomorrow at 2pm project sync",
                    tool_meta={"access_token": "token-1"},
                )
            )

        self.assertEqual(result.items[0].title, "[Created] Project Sync")
        quick_add_mock.assert_called_once()
        list_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
