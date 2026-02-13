import json
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

from cortexagent.config import settings
from .base import Tool, ToolContext, ToolResult, ToolResultItem


class WebSearchProvider:
    def search(self, query: str, max_results: int, timeout_seconds: int) -> list[ToolResultItem]:
        raise NotImplementedError


class BraveSearchProvider(WebSearchProvider):
    SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def search(self, query: str, max_results: int, timeout_seconds: int) -> list[ToolResultItem]:
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key,
        }
        params = {
            "q": query,
            "count": max_results,
            "text_decorations": False,
        }
        url = f"{self.SEARCH_URL}?{urlparse.urlencode(params)}"
        req = urlrequest.Request(url, headers=headers, method="GET")
        try:
            with urlrequest.urlopen(req, timeout=timeout_seconds) as res:
                payload = json.loads(res.read().decode("utf-8"))
        except urlerror.HTTPError as exc:
            raise RuntimeError(f"Brave search failed ({exc.code})")
        except Exception as exc:
            raise RuntimeError(f"Brave search failed: {exc}")

        raw_results = payload.get("web", {}).get("results", [])
        items: list[ToolResultItem] = []
        for row in raw_results:
            title = (row.get("title") or "").strip()
            url = (row.get("url") or "").strip()
            snippet = (row.get("description") or "").strip()
            if not title or not url:
                continue
            items.append(ToolResultItem(title=title, url=url, snippet=snippet))
        return items


class DuckDuckGoProvider(WebSearchProvider):
    SEARCH_URL = "https://api.duckduckgo.com/"

    def search(self, query: str, max_results: int, timeout_seconds: int) -> list[ToolResultItem]:
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "no_redirect": "1",
        }
        url = f"{self.SEARCH_URL}?{urlparse.urlencode(params)}"
        req = urlrequest.Request(url, method="GET")
        try:
            with urlrequest.urlopen(req, timeout=timeout_seconds) as res:
                payload = json.loads(res.read().decode("utf-8"))
        except urlerror.HTTPError as exc:
            raise RuntimeError(f"DuckDuckGo search failed ({exc.code})")
        except Exception as exc:
            raise RuntimeError(f"DuckDuckGo search failed: {exc}")

        items: list[ToolResultItem] = []
        abstract_text = (payload.get("AbstractText") or "").strip()
        abstract_url = (payload.get("AbstractURL") or "").strip()
        heading = (payload.get("Heading") or query).strip()

        if abstract_text and abstract_url:
            items.append(
                ToolResultItem(
                    title=heading,
                    url=abstract_url,
                    snippet=abstract_text,
                )
            )

        related = payload.get("RelatedTopics") or []
        for topic in related:
            if len(items) >= max_results:
                break
            text = (topic.get("Text") or "").strip()
            first_url = (topic.get("FirstURL") or "").strip()
            if not text or not first_url:
                continue
            title = text.split(" - ", 1)[0].strip() or "Related"
            items.append(ToolResultItem(title=title, url=first_url, snippet=text))

        return items[:max_results]


class WebSearchTool(Tool):
    name = "web_search"

    def __init__(self) -> None:
        provider_name = settings.web_search_provider.strip().lower()
        if provider_name == "brave":
            if not settings.brave_search_api_key:
                raise ValueError("BRAVE_SEARCH_API_KEY is required when WEB_SEARCH_PROVIDER=brave")
            self.provider: WebSearchProvider = BraveSearchProvider(settings.brave_search_api_key)
        elif provider_name == "duckduckgo":
            self.provider = DuckDuckGoProvider()
        else:
            raise ValueError(f"Unsupported WEB_SEARCH_PROVIDER '{settings.web_search_provider}'")

    def run(self, context: ToolContext) -> ToolResult:
        query = context.user_text.strip()
        items = self.provider.search(
            query=query,
            max_results=settings.web_search_max_results,
            timeout_seconds=settings.web_search_timeout_seconds,
        )
        return ToolResult(tool_name=self.name, query=query, items=items)
