import html
import json
import re
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

from cortexagent.config import settings
from .base import Tool, ToolContext, ToolResult, ToolResultItem

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


class WebSearchProvider:
    name = "unknown"

    def search(self, query: str, max_results: int, timeout_seconds: int) -> list[ToolResultItem]:
        raise NotImplementedError


class BraveSearchProvider(WebSearchProvider):
    name = "brave"
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
            title = _clean_html((row.get("title") or "").strip())
            url = _normalize_http_url((row.get("url") or "").strip())
            snippet = _clean_html((row.get("description") or "").strip())
            if not title or not url:
                continue
            items.append(ToolResultItem(title=title, url=url, snippet=snippet or title))
        return items


class DuckDuckGoProvider(WebSearchProvider):
    name = "duckduckgo"
    SEARCH_URL = "https://api.duckduckgo.com/"
    HTML_SEARCH_URL = "https://html.duckduckgo.com/html/"
    LITE_SEARCH_URL = "https://lite.duckduckgo.com/lite/"

    def search(self, query: str, max_results: int, timeout_seconds: int) -> list[ToolResultItem]:
        items: list[ToolResultItem] = []
        seen_urls: set[str] = set()
        cap = max(1, max_results)

        try:
            instant_items = self._instant_answer_search(query, cap, timeout_seconds)
            _append_unique_items(instant_items, items, seen_urls, cap)
        except Exception:
            pass
        if len(items) >= cap:
            return items[:cap]

        html_items = self._html_search(query=query, max_results=cap, timeout_seconds=timeout_seconds)
        _append_unique_items(html_items, items, seen_urls, cap)
        if len(items) >= cap:
            return items[:cap]

        lite_items = self._lite_search(query=query, max_results=cap, timeout_seconds=timeout_seconds)
        _append_unique_items(lite_items, items, seen_urls, cap)
        return items[:cap]

    def _instant_answer_search(
        self, query: str, max_results: int, timeout_seconds: int
    ) -> list[ToolResultItem]:
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "no_redirect": "1",
        }
        url = f"{self.SEARCH_URL}?{urlparse.urlencode(params)}"
        req = urlrequest.Request(
            url,
            headers={"User-Agent": DEFAULT_USER_AGENT, "Accept-Language": "en-US,en;q=0.8"},
            method="GET",
        )
        try:
            with urlrequest.urlopen(req, timeout=timeout_seconds) as res:
                payload = json.loads(res.read().decode("utf-8"))
        except urlerror.HTTPError as exc:
            raise RuntimeError(f"DuckDuckGo search failed ({exc.code})")
        except Exception as exc:
            raise RuntimeError(f"DuckDuckGo search failed: {exc}")

        items: list[ToolResultItem] = []
        abstract_text = _clean_html((payload.get("AbstractText") or "").strip())
        abstract_url = _normalize_http_url((payload.get("AbstractURL") or "").strip())
        heading = _clean_html((payload.get("Heading") or query).strip())

        if abstract_text and abstract_url:
            items.append(ToolResultItem(title=heading or query, url=abstract_url, snippet=abstract_text))

        related = payload.get("RelatedTopics") or []
        for topic in _iter_duck_related_topics(related):
            if len(items) >= max_results:
                break
            text = _clean_html((topic.get("Text") or "").strip())
            first_url = _normalize_http_url((topic.get("FirstURL") or "").strip())
            if not text or not first_url:
                continue
            title = text.split(" - ", 1)[0].strip() or "Related"
            items.append(ToolResultItem(title=title, url=first_url, snippet=text))

        return items[:max_results]

    def _html_search(
        self, query: str, max_results: int, timeout_seconds: int
    ) -> list[ToolResultItem]:
        params = {"q": query}
        url = f"{self.HTML_SEARCH_URL}?{urlparse.urlencode(params)}"
        req = urlrequest.Request(
            url,
            headers={"User-Agent": DEFAULT_USER_AGENT, "Accept-Language": "en-US,en;q=0.8"},
            method="GET",
        )
        try:
            with urlrequest.urlopen(req, timeout=timeout_seconds) as res:
                page = res.read().decode("utf-8", errors="ignore")
        except urlerror.HTTPError as exc:
            raise RuntimeError(f"DuckDuckGo HTML search failed ({exc.code})")
        except Exception as exc:
            raise RuntimeError(f"DuckDuckGo HTML search failed: {exc}")

        title_matches = list(
            re.finditer(
                r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                page,
                flags=re.IGNORECASE | re.DOTALL,
            )
        )
        snippet_matches = re.findall(
            r'<(?:a|div|span)[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div|span)>',
            page,
            flags=re.IGNORECASE | re.DOTALL,
        )

        items: list[ToolResultItem] = []
        for idx, match in enumerate(title_matches):
            if len(items) >= max_results:
                break
            raw_href = match.group(1).strip()
            title = _clean_html(match.group(2))
            if not raw_href or not title:
                continue
            url = _extract_duckduckgo_redirect_target(raw_href) or raw_href
            normalized_url = _normalize_http_url(url)
            if not normalized_url:
                continue
            snippet = _clean_html(snippet_matches[idx]) if idx < len(snippet_matches) else ""
            items.append(ToolResultItem(title=title, url=normalized_url, snippet=snippet or title))

        return items

    def _lite_search(
        self, query: str, max_results: int, timeout_seconds: int
    ) -> list[ToolResultItem]:
        params = {"q": query}
        url = f"{self.LITE_SEARCH_URL}?{urlparse.urlencode(params)}"
        req = urlrequest.Request(
            url,
            headers={"User-Agent": DEFAULT_USER_AGENT, "Accept-Language": "en-US,en;q=0.8"},
            method="GET",
        )
        try:
            with urlrequest.urlopen(req, timeout=timeout_seconds) as res:
                page = res.read().decode("utf-8", errors="ignore")
        except urlerror.HTTPError as exc:
            raise RuntimeError(f"DuckDuckGo lite search failed ({exc.code})")
        except Exception as exc:
            raise RuntimeError(f"DuckDuckGo lite search failed: {exc}")

        title_matches = list(
            re.finditer(
                r'<a[^>]*class="result-link"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                page,
                flags=re.IGNORECASE | re.DOTALL,
            )
        )
        snippet_matches = re.findall(
            r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>',
            page,
            flags=re.IGNORECASE | re.DOTALL,
        )

        items: list[ToolResultItem] = []
        for idx, match in enumerate(title_matches):
            if len(items) >= max_results:
                break
            raw_href = match.group(1).strip()
            title = _clean_html(match.group(2))
            if not raw_href or not title:
                continue
            url = _extract_duckduckgo_redirect_target(raw_href) or raw_href
            normalized_url = _normalize_http_url(url)
            if not normalized_url:
                continue
            snippet = _clean_html(snippet_matches[idx]) if idx < len(snippet_matches) else ""
            items.append(ToolResultItem(title=title, url=normalized_url, snippet=snippet or title))
        return items


class BingHtmlProvider(WebSearchProvider):
    name = "bing"
    SEARCH_URL = "https://www.bing.com/search"

    def search(self, query: str, max_results: int, timeout_seconds: int) -> list[ToolResultItem]:
        params = {"q": query, "setlang": "en-us", "ensearch": "1"}
        url = f"{self.SEARCH_URL}?{urlparse.urlencode(params)}"
        req = urlrequest.Request(
            url,
            headers={"User-Agent": DEFAULT_USER_AGENT, "Accept-Language": "en-US,en;q=0.8"},
            method="GET",
        )
        try:
            with urlrequest.urlopen(req, timeout=timeout_seconds) as res:
                page = res.read().decode("utf-8", errors="ignore")
        except urlerror.HTTPError as exc:
            raise RuntimeError(f"Bing search failed ({exc.code})")
        except Exception as exc:
            raise RuntimeError(f"Bing search failed: {exc}")

        blocks = re.findall(
            r'<li[^>]*class="[^"]*\bb_algo\b[^"]*"[^>]*>(.*?)</li>',
            page,
            flags=re.IGNORECASE | re.DOTALL,
        )
        items: list[ToolResultItem] = []

        for block in blocks:
            if len(items) >= max_results:
                break
            link_match = re.search(
                r"<h2[^>]*>\s*<a[^>]*href=\"([^\"]+)\"[^>]*>(.*?)</a>\s*</h2>",
                block,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if not link_match:
                continue
            raw_url = link_match.group(1).strip()
            title = _clean_html(link_match.group(2))
            if not title:
                continue
            url = _extract_bing_redirect_target(raw_url) or raw_url
            normalized_url = _normalize_http_url(url)
            if not normalized_url:
                continue

            snippet_match = re.search(r"<p[^>]*>(.*?)</p>", block, flags=re.IGNORECASE | re.DOTALL)
            snippet = _clean_html(snippet_match.group(1)) if snippet_match else ""
            items.append(ToolResultItem(title=title, url=normalized_url, snippet=snippet or title))

        return items


class WebSearchTool(Tool):
    name = "web_search"

    def __init__(self) -> None:
        self.providers = _build_provider_chain(
            configured=settings.web_search_provider,
            brave_key=settings.brave_search_api_key,
        )
        if not self.providers:
            raise ValueError("No usable web search provider configured.")

    def run(self, context: ToolContext) -> ToolResult:
        query = context.user_text.strip()
        if not query:
            return ToolResult(tool_name=self.name, query=query, items=[])

        max_results = max(1, settings.web_search_max_results)
        timeout_seconds = max(1, settings.web_search_timeout_seconds)
        retries = max(1, settings.web_search_retries)

        errors: list[str] = []
        items: list[ToolResultItem] = []
        seen_urls: set[str] = set()
        used_providers: list[str] = []

        for provider in self.providers:
            provider_items: list[ToolResultItem] = []
            for attempt in range(1, retries + 1):
                try:
                    provider_items = provider.search(
                        query=query,
                        max_results=max_results,
                        timeout_seconds=timeout_seconds,
                    )
                except Exception as exc:
                    errors.append(f"{provider.name} attempt {attempt}: {exc}")
                    continue

                if provider_items:
                    break
                errors.append(f"{provider.name} attempt {attempt}: empty_results")

            if not provider_items:
                continue

            before_len = len(items)
            _append_unique_items(provider_items, items, seen_urls, max_results)
            if len(items) > before_len:
                used_providers.append(provider.name)

            if len(items) >= max_results:
                break

        if items:
            provider_label = ",".join(used_providers) if used_providers else "mixed"
            return ToolResult(
                tool_name=f"{self.name}:{provider_label}",
                query=query,
                items=items[:max_results],
            )

        reason = "; ".join(errors[-6:]) if errors else "unknown"
        raise RuntimeError(f"Web search providers failed: {reason}")


def _iter_duck_related_topics(related: object) -> list[dict]:
    if not isinstance(related, list):
        return []
    out: list[dict] = []
    for row in related:
        if not isinstance(row, dict):
            continue
        topics = row.get("Topics")
        if isinstance(topics, list):
            for nested in topics:
                if isinstance(nested, dict):
                    out.append(nested)
            continue
        out.append(row)
    return out


def _append_unique_items(
    incoming: list[ToolResultItem],
    out: list[ToolResultItem],
    seen_urls: set[str],
    max_results: int,
) -> None:
    for item in incoming:
        if len(out) >= max_results:
            break
        normalized_url = _normalize_http_url(item.url)
        if not normalized_url:
            continue
        canonical_url = _canonicalize_url(normalized_url)
        if not canonical_url or canonical_url in seen_urls:
            continue
        seen_urls.add(canonical_url)
        out.append(
            ToolResultItem(
                title=_clean_html(item.title) or normalized_url,
                url=normalized_url,
                snippet=_clean_html(item.snippet) or _clean_html(item.title) or normalized_url,
            )
        )


def _clean_html(raw: str) -> str:
    no_tags = re.sub(r"<[^>]+>", " ", raw or "")
    compact = html.unescape(no_tags).replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", compact).strip()


def _normalize_http_url(raw_url: str) -> str:
    cleaned = (raw_url or "").strip()
    if not cleaned:
        return ""
    parsed = urlparse.urlparse(cleaned)
    if parsed.scheme.lower() not in {"http", "https"}:
        return ""
    if not parsed.netloc:
        return ""
    return cleaned


def _canonicalize_url(raw_url: str) -> str:
    parsed = urlparse.urlparse(raw_url)
    if not parsed.scheme or not parsed.netloc:
        return ""
    path = parsed.path or "/"
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{path}{query}"


def _extract_duckduckgo_redirect_target(raw_href: str) -> str | None:
    parsed = urlparse.urlparse(raw_href)
    query = urlparse.parse_qs(parsed.query)
    target = query.get("uddg", [None])[0]
    if not target:
        return None
    return urlparse.unquote(target).strip()


def _extract_bing_redirect_target(raw_href: str) -> str | None:
    parsed = urlparse.urlparse(raw_href)
    if "bing.com" not in parsed.netloc.lower():
        return None
    query = urlparse.parse_qs(parsed.query)
    token = query.get("u", [None])[0]
    if not token:
        return None
    decoded = urlparse.unquote(token).strip()
    if decoded.startswith("a1"):
        decoded = decoded[2:]
    normalized = _normalize_http_url(decoded)
    return normalized or None


def _build_provider_chain(
    configured: str, brave_key: str | None
) -> list[WebSearchProvider]:
    tokens = [
        token.strip().lower()
        for token in configured.split(",")
        if token.strip()
    ]
    if not tokens:
        tokens = ["duckduckgo", "bing"]

    providers: list[WebSearchProvider] = []
    seen: set[str] = set()

    for token in tokens:
        if token in seen:
            continue
        if token == "duckduckgo":
            providers.append(DuckDuckGoProvider())
            seen.add(token)
            continue
        if token == "bing":
            providers.append(BingHtmlProvider())
            seen.add(token)
            continue
        if token == "brave":
            if not brave_key:
                continue
            providers.append(BraveSearchProvider(brave_key))
            seen.add(token)
            continue
        raise ValueError(f"Unsupported WEB_SEARCH_PROVIDER '{token}'")

    if "duckduckgo" not in seen:
        providers.append(DuckDuckGoProvider())
    if "bing" not in seen:
        providers.append(BingHtmlProvider())

    return providers
