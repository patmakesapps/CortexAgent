from dataclasses import dataclass

from cortexagent.config import settings
from cortexagent.models import AgentDecision
from cortexagent.router import decide_action
from cortexagent.services.cortexltm_client import CortexLTMClient
from cortexagent.tools import ToolContext, ToolRegistry


@dataclass(frozen=True)
class OrchestratorResult:
    response: str
    decision: AgentDecision
    sources: list[dict[str, str]]


class AgentOrchestrator:
    def __init__(self, ltm_client: CortexLTMClient, tool_registry: ToolRegistry) -> None:
        self.ltm_client = ltm_client
        self.tool_registry = tool_registry

    def handle_chat(
        self,
        thread_id: str,
        text: str,
        short_term_limit: int | None,
        authorization: str | None,
    ) -> OrchestratorResult:
        route = decide_action(
            user_text=text,
            tools_enabled=settings.agent_tools_enabled,
            web_search_enabled=settings.web_search_enabled,
        )

        if route.action != "web_search":
            assistant_text = self.ltm_client.chat(
                thread_id=thread_id,
                text=text,
                short_term_limit=short_term_limit,
                authorization=authorization,
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
        result = tool.run(ToolContext(thread_id=thread_id, user_text=text))
        assistant_text, sources = _format_web_search_response(result.items)

        self.ltm_client.add_event(
            thread_id=thread_id,
            actor="user",
            content=text,
            meta={"source": "cortexagent", "decision": route.action},
            authorization=authorization,
        )
        self.ltm_client.add_event(
            thread_id=thread_id,
            actor="assistant",
            content=assistant_text,
            meta={
                "source": "cortexagent_web_search",
                "tool": result.tool_name,
                "query": result.query,
                "source_urls": [s["url"] for s in sources],
            },
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


def _format_web_search_response(items: list) -> tuple[str, list[dict[str, str]]]:
    if not items:
        return (
            "I could not find reliable web results for that query right now.",
            [],
        )

    lines = ["Here is what I found from web sources:"]
    sources: list[dict[str, str]] = []
    for idx, item in enumerate(items, start=1):
        snippet = item.snippet.strip()
        lines.append(f"{idx}. {item.title}: {snippet}")
        sources.append({"title": item.title, "url": item.url})

    lines.append("Sources:")
    for src in sources:
        lines.append(f"- {src['title']}: {src['url']}")

    return "\n".join(lines), sources
