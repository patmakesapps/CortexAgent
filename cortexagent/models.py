from pydantic import BaseModel, Field


class AgentChatRequest(BaseModel):
    text: str = Field(min_length=1, max_length=6000)
    short_term_limit: int | None = Field(default=30, ge=1, le=200)


class AgentDecision(BaseModel):
    action: str
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class AgentChatResponse(BaseModel):
    thread_id: str
    response: str
    decision: AgentDecision
    sources: list[dict[str, str]] = Field(default_factory=list)
