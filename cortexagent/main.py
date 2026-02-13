from fastapi import FastAPI, Header, HTTPException

from cortexagent.config import settings
from cortexagent.models import AgentChatRequest, AgentChatResponse
from cortexagent.services import AgentOrchestrator, CortexLTMClient
from cortexagent.tools import ToolRegistry, WebSearchTool

app = FastAPI(title="CortexAgent", version="0.1.0")


def _build_orchestrator() -> AgentOrchestrator:
    registry = ToolRegistry()
    if settings.web_search_enabled:
        registry.register(WebSearchTool())

    ltm_client = CortexLTMClient(
        base_url=settings.cortexltm_api_base_url,
        api_key=settings.cortexltm_api_key,
    )
    return AgentOrchestrator(ltm_client=ltm_client, tool_registry=registry)


orchestrator = _build_orchestrator()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/agent/threads/{thread_id}/chat", response_model=AgentChatResponse)
def agent_chat(
    thread_id: str,
    payload: AgentChatRequest,
    authorization: str | None = Header(default=None),
) -> AgentChatResponse:
    try:
        result = orchestrator.handle_chat(
            thread_id=thread_id,
            text=payload.text,
            short_term_limit=payload.short_term_limit,
            authorization=authorization,
        )
        return AgentChatResponse(
            thread_id=thread_id,
            response=result.response,
            decision=result.decision,
            sources=result.sources,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected agent error: {exc}")
