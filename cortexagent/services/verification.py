from dataclasses import dataclass
from datetime import datetime
import re
from urllib.parse import urlparse


HIGH_STAKES_CUES = {
    "medical",
    "medicine",
    "drug",
    "dosage",
    "symptom",
    "diagnosis",
    "legal",
    "law",
    "lawsuit",
    "regulation",
    "tax",
    "finance",
    "financial",
    "stock",
    "crypto",
    "price",
    "interest rate",
    "mortgage",
    "safety",
    "hazard",
    "security advisory",
    "vulnerability",
}

TEMPORAL_CUES = {
    "live",
    "latest",
    "current",
    "today",
    "now",
    "this week",
    "breaking",
    "recent",
    "updated",
}

NUMERIC_SENSITIVE_CUES = {
    "price",
    "cost",
    "rate",
    "percent",
    "%",
    "amount",
    "market cap",
    "revenue",
    "valuation",
    "quote",
}

SHOPPING_RESEARCH_CUES = {
    "buy",
    "purchase",
    "shopping",
    "recommend",
    "recommendation",
    "options",
    "models",
    "best",
    "under ",
    "budget",
    "price range",
}

CRITICAL_HIGH_STAKES_CUES = {
    "medical",
    "diagnosis",
    "treatment",
    "prescription",
    "legal",
    "law",
    "tax",
    "irs",
    "security advisory",
    "vulnerability",
    "hazard",
}


@dataclass(frozen=True)
class VerificationProfile:
    level: str
    reasons: list[str]
    requires_web_verification: bool
    min_independent_sources: int


def assess_verification_profile(user_text: str) -> VerificationProfile:
    text = re.sub(r"\s+", " ", user_text.strip().lower())
    reasons: list[str] = []

    if _looks_like_shopping_research(text) and not _contains_any_cue(text, CRITICAL_HIGH_STAKES_CUES):
        shopping_reasons = ["shopping_research"]
        if _contains_any_cue(text, TEMPORAL_CUES):
            shopping_reasons.append("time_sensitive")
        return VerificationProfile(
            level="medium",
            reasons=shopping_reasons,
            requires_web_verification=True,
            min_independent_sources=1,
        )

    if _contains_any_cue(text, HIGH_STAKES_CUES):
        reasons.append("high_stakes")
    if _contains_any_cue(text, TEMPORAL_CUES):
        reasons.append("time_sensitive")

    if "high_stakes" in reasons or ("time_sensitive" in reasons and _looks_factual(text)):
        return VerificationProfile(
            level="high",
            reasons=reasons or ["factual_query"],
            requires_web_verification=True,
            min_independent_sources=2,
        )
    if "time_sensitive" in reasons:
        return VerificationProfile(
            level="medium",
            reasons=reasons,
            requires_web_verification=True,
            min_independent_sources=1,
        )
    return VerificationProfile(
        level="low",
        reasons=[],
        requires_web_verification=False,
        min_independent_sources=0,
    )


def enforce_verification_policy(
    user_text: str,
    assistant_text: str,
    sources: list[dict[str, str]],
    profile: VerificationProfile,
) -> str:
    if profile.level != "high":
        return assistant_text

    independent_count = _count_independent_sources(sources)
    if independent_count < profile.min_independent_sources:
        base = (
            "I cannot verify this high-risk request with enough independent sources yet. "
            "Please retry in a moment or check authoritative primary sources directly."
        )
        return _append_sources(base, sources)

    if _is_numeric_sensitive(user_text):
        response_values = _extract_money_values(assistant_text)
        source_values = _extract_money_values_from_sources(sources)
        if response_values and not source_values:
            base = (
                "I cannot verify the numeric value with reliable source evidence right now. "
                "Please check the linked primary sources directly."
            )
            return _append_sources(base, sources)
        if response_values and source_values and _has_money_mismatch(response_values, source_values):
            base = (
                "I found conflicting numeric values across sources, so I cannot provide a single "
                "verified number right now. Please use the linked primary sources."
            )
            return _append_sources(base, sources)

    stamp = _friendly_local_timestamp()
    return f"As of {stamp}, verified against {independent_count} independent sources.\n{assistant_text}"


def _append_sources(base_text: str, sources: list[dict[str, str]]) -> str:
    lines = [base_text]
    if not sources:
        return "\n".join(lines)
    lines.append("")
    lines.append("Sources:")
    for src in sources[:5]:
        title = (src.get("title", "") or "").strip() or "Source"
        url = (src.get("url", "") or "").strip()
        if not url:
            continue
        lines.append(f"- {title}: {url}")
    return "\n".join(lines)


def _looks_factual(text: str) -> bool:
    factual_cues = {"what is", "who is", "when is", "how much", "how many", "price", "rate"}
    return any(cue in text for cue in factual_cues)


def _is_numeric_sensitive(text: str) -> bool:
    lowered = text.lower()
    return _contains_any_cue(lowered, NUMERIC_SENSITIVE_CUES)


def _looks_like_shopping_research(text: str) -> bool:
    if _contains_any_cue(text, SHOPPING_RESEARCH_CUES):
        return True
    return bool(
        re.search(
            r"\b(boat|tractor|car|truck|suv|motorcycle|laptop|phone|camera|tv|appliance|sofa|mower)\b",
            text,
        )
        and re.search(r"\b(for|under|budget|people|use|need)\b", text)
    )


def _contains_any_cue(text: str, cues: set[str]) -> bool:
    return any(_contains_cue(text, cue) for cue in cues)


def _contains_cue(text: str, cue: str) -> bool:
    return bool(re.search(rf"\b{re.escape(cue.strip())}\b", text))


def _count_independent_sources(sources: list[dict[str, str]]) -> int:
    hosts: set[str] = set()
    for src in sources:
        url = src.get("url", "")
        host = (urlparse(url).netloc or "").lower()
        host = host.removeprefix("www.")
        if host:
            hosts.add(host)
    return len(hosts)


def _extract_money_values_from_sources(sources: list[dict[str, str]]) -> list[float]:
    values: list[float] = []
    for src in sources:
        values.extend(_extract_money_values(src.get("title", "")))
        values.extend(_extract_money_values(src.get("snippet", "")))
    return values


def _extract_money_values(text: str) -> list[float]:
    pattern = re.compile(
        r"(?i)(?:\$\s*|usd\s*)([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)"
    )
    out: list[float] = []
    for match in pattern.finditer(text):
        raw = match.group(1).replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        if value <= 0 or value > 1_000_000_000:
            continue
        out.append(value)
    return out


def _has_money_mismatch(response_values: list[float], source_values: list[float]) -> bool:
    tolerance_pct = 0.03
    tolerance_abs = 3.0
    for response_value in response_values:
        nearest = min(source_values, key=lambda source_value: abs(source_value - response_value))
        diff = abs(nearest - response_value)
        if diff <= tolerance_abs:
            continue
        if diff / max(nearest, 1.0) <= tolerance_pct:
            continue
        return True
    return False


def _friendly_local_timestamp() -> str:
    local_now = datetime.now().astimezone()
    return local_now.strftime("%b %d, %Y at %I:%M %p %Z").replace(" at 0", " at ")
