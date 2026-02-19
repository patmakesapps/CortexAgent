from __future__ import annotations

import json
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

from .base import Tool, ToolContext, ToolResult, ToolResultItem


class GoogleDriveTool(Tool):
    name = "google_drive"
    DRIVE_FILES_URL = "https://www.googleapis.com/drive/v3/files"

    def run(self, context: ToolContext) -> ToolResult:
        tool_meta = context.tool_meta or {}
        access_token = tool_meta.get("access_token")
        if not isinstance(access_token, str) or not access_token.strip():
            raise RuntimeError("Google account is not connected. Please connect Google first.")

        tool_meta = context.tool_meta if isinstance(context.tool_meta, dict) else {}
        args = tool_meta.get("args") if isinstance(tool_meta.get("args"), dict) else {}
        user_text = (context.user_text or "").strip()
        max_results = 8
        query = str((args or {}).get("query") or "").strip() or None
        items = self._list_files(
            access_token=access_token.strip(),
            max_results=max_results,
            query=query,
        )
        return ToolResult(tool_name=self.name, query=user_text, items=items)

    def _list_files(
        self,
        access_token: str,
        max_results: int,
        query: str | None,
    ) -> list[ToolResultItem]:
        params = {
            "pageSize": str(max_results),
            "fields": "files(id,name,mimeType,modifiedTime,webViewLink,owners(displayName))",
            "orderBy": "modifiedTime desc",
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
        }
        if query:
            safe_query = query.replace("'", "\\'")
            params["q"] = (
                f"trashed = false and "
                f"(name contains '{safe_query}' or fullText contains '{safe_query}')"
            )
        else:
            params["q"] = "trashed = false"
        url = f"{self.DRIVE_FILES_URL}?{urlparse.urlencode(params)}"

        req = urlrequest.Request(
            url,
            method="GET",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {access_token}",
            },
        )
        try:
            with urlrequest.urlopen(req, timeout=8) as res:
                payload = json.loads(res.read().decode("utf-8"))
        except urlerror.HTTPError as exc:
            if exc.code in {401, 403}:
                raise RuntimeError("Google Drive authorization failed. Please reconnect Google.")
            raise RuntimeError(f"Google Drive API failed ({exc.code}).")
        except Exception as exc:
            raise RuntimeError(f"Google Drive API failed: {exc}")

        rows = payload.get("files", []) if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            return []

        out: list[ToolResultItem] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            file_id = str(row.get("id") or "").strip()
            file_name = str(row.get("name") or "Untitled").strip()
            mime_type = str(row.get("mimeType") or "").strip()
            modified = str(row.get("modifiedTime") or "").strip()
            owners = row.get("owners")
            owner_name = ""
            if isinstance(owners, list) and owners and isinstance(owners[0], dict):
                owner_name = str(owners[0].get("displayName") or "").strip()
            snippet_parts = [f"Type: {_friendly_type(mime_type)}"]
            if owner_name:
                snippet_parts.append(f"Owner: {owner_name}")
            if modified:
                snippet_parts.append(f"Updated: {modified}")
            out.append(
                ToolResultItem(
                    title=file_name,
                    url=str(row.get("webViewLink") or _default_drive_url(file_id)).strip(),
                    snippet=" | ".join(snippet_parts),
                )
            )
        return out


def _friendly_type(mime_type: str) -> str:
    lowered = (mime_type or "").lower()
    if lowered == "application/vnd.google-apps.document":
        return "Google Doc"
    if lowered == "application/vnd.google-apps.spreadsheet":
        return "Google Sheet"
    if lowered == "application/vnd.google-apps.presentation":
        return "Google Slides"
    if lowered == "application/vnd.google-apps.folder":
        return "Folder"
    if lowered.startswith("application/vnd.google-apps"):
        return "Google File"
    if lowered:
        return lowered
    return "Unknown"


def _default_drive_url(file_id: str) -> str:
    if not file_id:
        return "https://drive.google.com/drive/my-drive"
    return f"https://drive.google.com/file/d/{file_id}/view"
