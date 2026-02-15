import unittest
from unittest.mock import patch

from cortexagent.tools.base import ToolContext, ToolResultItem
from cortexagent.tools.google_drive import GoogleDriveTool


class GoogleDriveToolTests(unittest.TestCase):
    def test_requires_access_token(self):
        tool = GoogleDriveTool()
        with self.assertRaises(RuntimeError):
            tool.run(ToolContext(thread_id="t1", user_text="find file quarterly plan"))

    def test_lists_drive_files(self):
        tool = GoogleDriveTool()
        with patch.object(
            tool,
            "_list_files",
            return_value=[
                ToolResultItem(
                    title="Q1 Plan",
                    url="https://drive.google.com/file/d/abc/view",
                    snippet="Type: Google Doc | Owner: Pat",
                )
            ],
        ):
            result = tool.run(
                ToolContext(
                    thread_id="t1",
                    user_text="find file quarterly plan",
                    tool_meta={"access_token": "token-1"},
                )
            )
        self.assertEqual(result.tool_name, "google_drive")
        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.items[0].title, "Q1 Plan")


if __name__ == "__main__":
    unittest.main()
