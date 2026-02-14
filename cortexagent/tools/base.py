from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolContext:
    thread_id: str
    user_text: str
    tool_meta: dict[str, object] | None = None


@dataclass(frozen=True)
class ToolResultItem:
    title: str
    url: str
    snippet: str


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    query: str
    items: list[ToolResultItem]


class Tool(ABC):
    name: str

    @abstractmethod
    def run(self, context: ToolContext) -> ToolResult:
        raise NotImplementedError
