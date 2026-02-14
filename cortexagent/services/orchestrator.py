from dataclasses import dataclass
from datetime import datetime
import re
import json
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib import request as urlrequest

from cortexagent.config import settings
from cortexagent.models import AgentDecision
from cortexagent.router import RouteDecision, decide_action
from cortexagent.services.cortexltm_client import CortexLTMClient
from cortexagent.services.verification import (
    assess_verification_profile,
    enforce_verification_policy,
)
from cortexagent.tools import ToolContext, ToolRegistry


@dataclass(frozen=True)
class OrchestratorResult:
    response: str
    decision: AgentDecision
    sources: list[dict[str, str]]


class AgentOrchestrator:
    def __init__(
        self, ltm_client: CortexLTMClient, tool_registry: ToolRegistry
    ) -> None:
        self.ltm_client = ltm_client
        self.tool_registry = tool_registry

    def handle_chat(
        self,
        thread_id: str,
        text: str,
        short_term_limit: int | None,
        authorization: str | None,
    ) -> OrchestratorResult:
        verification = assess_verification_profile(text)
        route = decide_action(
            user_text=text,
            tools_enabled=settings.agent_tools_enabled,
            web_search_enabled=settings.web_search_enabled,
        )
        if (
            verification.requires_web_verification
            and settings.agent_tools_enabled
            and settings.web_search_enabled
            and route.action != "web_search"
        ):
            route = RouteDecision(
                action="web_search",
                reason=f"verification_override:{','.join(verification.reasons)}",
                confidence=max(route.confidence, 0.9),
            )

        if route.action != "web_search":
            assistant_text = self.ltm_client.chat(
                thread_id=thread_id,
                text=text,
                short_term_limit=short_term_limit,
                authorization=authorization,
            )
            assistant_text = enforce_verification_policy(
                user_text=text,
                assistant_text=assistant_text,
                sources=[],
                profile=verification,
            )
            return OrchestratorResult(
                response=assistant_text,
                decision=AgentDecision(
                    action=route.action,
                    reason=route.reason,
                    confidence=route.confidence,
                ),
                sources=[],
            )

        tool = self.tool_registry.get("web_search")
        try:
            result = tool.run(ToolContext(thread_id=thread_id, user_text=text))
        except Exception as exc:
            assistant_text = (
                "I routed this request to web search, but the search providers failed. "
                f"Error: {exc}"
            )
            return OrchestratorResult(
                response=assistant_text,
                decision=AgentDecision(
                    action="web_search",
                    reason="web_search_failed",
                    confidence=1.0,
                ),
                sources=[],
            )

        assistant_text, sources = _format_web_search_response(result.items, text)
        assistant_text = _verify_numeric_claims(
            user_text=text,
            assistant_text=assistant_text,
            sources=sources,
        )
        assistant_text = enforce_verification_policy(
            user_text=text,
            assistant_text=assistant_text,
            sources=sources,
            profile=verification,
        )
        self._persist_web_search_events(
            thread_id=thread_id,
            user_text=text,
            assistant_text=assistant_text,
            tool_name=result.tool_name,
            query=result.query,
            sources=sources,
            authorization=authorization,
        )

        return OrchestratorResult(
            response=assistant_text,
            decision=AgentDecision(
                action=route.action,
                reason=route.reason,
                confidence=route.confidence,
            ),
            sources=sources,
        )

    def _persist_web_search_events(
        self,
        thread_id: str,
        user_text: str,
        assistant_text: str,
        tool_name: str,
        query: str,
        sources: list[dict[str, str]],
        authorization: str | None,
    ) -> None:
        # Best effort persistence: tool responses should still return even if writes fail.
        try:
            self.ltm_client.add_event(
                thread_id=thread_id,
                actor="user",
                content=user_text,
                meta={"source": "cortexagent", "decision": "web_search"},
                authorization=authorization,
            )
        except Exception:
            pass

        try:
            self.ltm_client.add_event(
                thread_id=thread_id,
                actor="assistant",
                content=assistant_text,
                meta={
                    "source": "cortexagent_web_search",
                    "tool": tool_name,
                    "query": query,
                    "source_urls": [s["url"] for s in sources],
                },
                authorization=authorization,
            )
        except Exception:
            pass


def _format_web_search_response(
    items: list, user_text: str
) -> tuple[str, list[dict[str, str]]]:
    if not items:
        return (
            "I could not find reliable web results for that query right now.",
            [],
        )

    timestamp = _friendly_local_timestamp()
    raw_sources: list[dict[str, str]] = []
    for item in items[:8]:
        snippet = item.snippet.strip()
        raw_sources.append({"title": item.title, "url": item.url, "snippet": snippet})

    sources = _select_clean_sources(raw_sources, user_text=user_text, max_count=3)

    list_limit = 3
    if _looks_like_live_price_query(user_text):
        subject = _infer_price_subject(user_text)
        spot_price, spot_source = _fetch_crypto_spot_price_usd(user_text)
        values = _extract_money_values_from_sources(sources)
        if spot_price is not None:
            headline = f"As of {timestamp}, {subject} is about ${spot_price:,.2f} USD."
        elif values:
            low = min(values)
            high = max(values)
            mid = values[len(values) // 2]
            if (high - low) / max(mid, 1.0) <= 0.006:
                headline = f"As of {timestamp}, {subject} is about ${mid:,.2f} USD."
            else:
                headline = (
                    f"As of {timestamp}, {subject} appears between "
                    f"${low:,.2f} and ${high:,.2f} USD across sources."
                )
        else:
            headline = (
                f"As of {timestamp}, I could not extract a stable live {subject} price "
                "from the available snippets."
            )
        lines = [headline, "Sources:"]
        if spot_source:
            lines.append(f"- CoinGecko spot quote: {spot_source}")
            list_limit = 2
    else:
        lines = ["Here is what I found from web sources:"]
        for idx, src in enumerate(sources, start=1):
            snippet = src.get("snippet", "")
            title = src.get("title", "")
            lines.append(f"{idx}. {title}: {snippet}")
        lines.append("Sources:")

    for src in sources[:list_limit]:
        lines.append(f"- {src['title']}: {src['url']}")

    return "\n".join(lines), sources


PRICE_QUERY_TOKENS = {
    "price",
    "live price",
    "current price",
    "quote",
    "market price",
    "trading at",
    "usd",
    "eur",
}


def _verify_numeric_claims(
    user_text: str,
    assistant_text: str,
    sources: list[dict[str, str]],
) -> str:
    if not assistant_text.strip() or not sources:
        return assistant_text
    # Price responses are already rendered in a source-grounded template.
    # Avoid prepending a second correction block.
    if _looks_like_live_price_query(user_text):
        return assistant_text

    source_values = _extract_money_values_from_sources(sources)
    response_values = _extract_money_values(assistant_text)
    if not source_values or not response_values:
        return assistant_text
    if not _has_numeric_mismatch(response_values, source_values):
        return assistant_text

    low = min(source_values)
    high = max(source_values)
    mid = source_values[len(source_values) // 2]
    timestamp = _friendly_local_timestamp()

    if (high - low) / max(mid, 1.0) <= 0.006:
        corrected = f"As of {timestamp}, cited sources cluster around ${mid:,.2f}."
    else:
        corrected = (
            f"As of {timestamp}, cited sources vary between ${low:,.2f} and ${high:,.2f}."
        )

    return (
        f"{corrected} Source snippets disagree, so I am showing a range instead of a single "
        "exact live price.\n\n"
        f"{assistant_text}"
    )


def _looks_like_live_price_query(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return any(token in normalized for token in PRICE_QUERY_TOKENS)


def _infer_price_subject(text: str) -> str:
    lowered = text.strip().lower()
    known_assets = [
        "bitcoin",
        "btc",
        "ethereum",
        "eth",
        "xrp",
        "solana",
        "sol",
        "dogecoin",
        "doge",
        "cardano",
        "ada",
    ]
    for asset in known_assets:
        if re.search(rf"\b{re.escape(asset)}\b", lowered):
            return asset.upper() if len(asset) <= 4 else asset
    return "the asset"


_COINGECKO_ID_BY_ALIAS = {
    "btc": "bitcoin",
    "bitcoin": "bitcoin",
    "eth": "ethereum",
    "ethereum": "ethereum",
    "xrp": "ripple",
    "sol": "solana",
    "solana": "solana",
    "doge": "dogecoin",
    "dogecoin": "dogecoin",
    "ada": "cardano",
    "cardano": "cardano",
}

_COINBASE_PRODUCT_BY_ALIAS = {
    "btc": "BTC-USD",
    "bitcoin": "BTC-USD",
    "eth": "ETH-USD",
    "ethereum": "ETH-USD",
    "xrp": "XRP-USD",
    "sol": "SOL-USD",
    "solana": "SOL-USD",
    "doge": "DOGE-USD",
    "dogecoin": "DOGE-USD",
    "ada": "ADA-USD",
    "cardano": "ADA-USD",
}


def _fetch_crypto_spot_price_usd(user_text: str) -> tuple[float | None, str | None]:
    lowered = user_text.lower()
    alias = None
    for key in _COINGECKO_ID_BY_ALIAS.keys():
        if re.search(rf"\b{re.escape(key)}\b", lowered):
            alias = key
            break
    if not alias:
        return (None, None)

    coinbase_price = _fetch_coinbase_spot_price_usd(alias)
    if coinbase_price is not None:
        product = _COINBASE_PRODUCT_BY_ALIAS[alias]
        return (coinbase_price, f"https://www.coinbase.com/price/{product.split('-')[0].lower()}")

    coin_id = _COINGECKO_ID_BY_ALIAS[alias]
    url = (
        "https://api.coingecko.com/api/v3/simple/price"
        f"?ids={coin_id}&vs_currencies=usd"
    )
    req = urlrequest.Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urlrequest.urlopen(req, timeout=5) as res:
            payload = json.loads(res.read().decode("utf-8"))
    except Exception:
        return (None, None)

    row = payload.get(coin_id, {}) if isinstance(payload, dict) else {}
    value = row.get("usd") if isinstance(row, dict) else None
    if not isinstance(value, (int, float)) or value <= 0:
        return (None, None)
    source_url = f"https://www.coingecko.com/en/coins/{coin_id}"
    return (float(value), source_url)


def _fetch_coinbase_spot_price_usd(alias: str) -> float | None:
    product = _COINBASE_PRODUCT_BY_ALIAS.get(alias)
    if not product:
        return None
    url = f"https://api.coinbase.com/v2/prices/{product}/spot"
    req = urlrequest.Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urlrequest.urlopen(req, timeout=4) as res:
            payload = json.loads(res.read().decode("utf-8"))
    except Exception:
        return None
    data = payload.get("data", {}) if isinstance(payload, dict) else {}
    raw_amount = data.get("amount") if isinstance(data, dict) else None
    if not isinstance(raw_amount, str):
        return None
    try:
        value = float(raw_amount)
    except ValueError:
        return None
    return value if value > 0 else None


def _extract_money_values_from_sources(sources: list[dict[str, str]]) -> list[float]:
    values: list[float] = []
    for src in sources:
        snippet = src.get("snippet", "")
        title = src.get("title", "")
        values.extend(_extract_money_values(snippet))
        values.extend(_extract_money_values(title))
    values.sort()
    return values[:24]


def _extract_money_values(text: str) -> list[float]:
    # Extract likely *price* values; avoid volume/market-cap numbers.
    pattern = re.compile(
        r"(?i)(?:\$\s*|usd\s*)"
        r"([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)"
        r"(?:\s*usd)?"
    )
    blocked_context = {
        "volume",
        "24-hour",
        "24 hour",
        "market cap",
        "fully diluted",
        "circulating",
        "trading volume",
    }
    out: list[float] = []
    for match in pattern.finditer(text):
        raw = match.group(1).replace(",", "")
        start, end = match.span()
        window = text[max(0, start - 28) : min(len(text), end + 28)].lower()
        if any(token in window for token in blocked_context):
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if value <= 0:
            continue
        if value > 1_000_000:
            continue
        out.append(value)
    return out


def _has_numeric_mismatch(response_values: list[float], source_values: list[float]) -> bool:
    tolerance_pct = 0.02
    tolerance_abs = 2.0
    for response_value in response_values:
        nearest = min(source_values, key=lambda source_value: abs(source_value - response_value))
        diff = abs(nearest - response_value)
        if diff <= tolerance_abs:
            continue
        if diff / max(nearest, 1.0) <= tolerance_pct:
            continue
        return True
    return False


_TRACKING_KEYS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "msclkid",
    "gclid",
    "fbclid",
    "igshid",
    "ref",
    "ref_src",
    "source",
    "ad_domain",
    "ad_provider",
    "ad_type",
    "click_metadata",
    "rut",
    "u",
    "u3",
    "rlid",
    "vqd",
    "iurl",
    "cid",
    "id",
    "ig",
}

_PRICE_SOURCE_HOST_PREFERENCE = [
    "coinmarketcap.com",
    "coindesk.com",
    "google.com",
    "finance.yahoo.com",
    "coinbase.com",
]


def _select_clean_sources(
    sources: list[dict[str, str]], user_text: str, max_count: int
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen_hosts: set[str] = set()
    price_mode = _looks_like_live_price_query(user_text)

    ordered = sources
    if price_mode:
        ordered = sorted(sources, key=lambda s: _source_rank_for_price(s.get("url", "")))

    for src in ordered:
        raw_url = src.get("url", "")
        cleaned_url = _clean_source_url(raw_url)
        if not cleaned_url:
            continue
        host = (urlparse(cleaned_url).netloc or "").lower()
        if not host or host in seen_hosts:
            continue
        title = (src.get("title", "") or "").strip()
        if _looks_like_ad_source(title, cleaned_url):
            continue
        seen_hosts.add(host)
        out.append(
            {
                "title": title or host,
                "url": cleaned_url,
                "snippet": (src.get("snippet", "") or "").strip(),
            }
        )
        if len(out) >= max_count:
            break

    return out


def _source_rank_for_price(url: str) -> tuple[int, str]:
    host = (urlparse(url).netloc or "").lower()
    for idx, preferred in enumerate(_PRICE_SOURCE_HOST_PREFERENCE):
        if preferred in host:
            return (idx, host)
    return (len(_PRICE_SOURCE_HOST_PREFERENCE) + 1, host)


def _looks_like_ad_source(title: str, url: str) -> bool:
    lowered_title = title.lower()
    lowered_url = url.lower()
    ad_markers = [
        "trusted",
        "sign up",
        "buy now",
        "easy cryptocurrency trading",
        "advert",
    ]
    if any(marker in lowered_title for marker in ad_markers):
        return True
    if "duckduckgo.com/y.js" in lowered_url or "bing.com/aclick" in lowered_url:
        return True
    return False


def _clean_source_url(raw_url: str) -> str:
    parsed = urlparse((raw_url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""

    host = parsed.netloc.lower()
    path = parsed.path or "/"
    if ("duckduckgo.com" in host and path == "/y.js") or ("bing.com" in host and path == "/aclick"):
        return ""

    kept = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=False):
        if key.lower() in _TRACKING_KEYS:
            continue
        kept.append((key, value))

    query = urlencode(kept, doseq=True)
    cleaned = parsed._replace(query=query, fragment="")
    normalized = urlunparse(cleaned)
    return normalized


def _friendly_local_timestamp() -> str:
    local_now = datetime.now().astimezone()
    stamp = local_now.strftime("%b %d, %Y at %I:%M %p %Z")
    return stamp.replace(" at 0", " at ")
