from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cortexagent.services.planner import PlannedStep
from cortexagent.tools import ToolContext
from cortexagent.tools.registry import ToolRegistry


@dataclass(frozen=True)
class ExecutedStep:
    id: str
    action: str
    tool_name: str
    success: bool
    reason: str
    execution_status: str
    query: str
    capability_label: str
    items: list[dict[str, str]]
    error: str | None = None


class ToolExecutor:
    def __init__(self, tool_registry: ToolRegistry) -> None:
        self._tool_registry = tool_registry

    def execute_steps(
        self,
        *,
        thread_id: str,
        user_text: str,
        plan_steps: list[PlannedStep],
        tool_meta: dict[str, object],
    ) -> list[ExecutedStep]:
        out: list[ExecutedStep] = []
        for step in plan_steps:
            tool_def = self._tool_registry.get_definition(step.tool)
            validated_args = tool_def.validate_args(step.args)
            step_tool_meta = {
                **tool_meta,
                "operation": validated_args.get("operation", "read"),
                "args": validated_args,
            }
            try:
                result = tool_def.tool.run(
                    ToolContext(
                        thread_id=thread_id,
                        user_text=user_text,
                        tool_meta=step_tool_meta,
                    )
                )
                items = [
                    {
                        "title": item.title,
                        "url": item.url,
                        "snippet": item.snippet,
                    }
                    for item in result.items
                ]
                out.append(
                    ExecutedStep(
                        id=step.id,
                        action=tool_def.name,
                        tool_name=tool_def.name,
                        success=True,
                        reason=step.reason,
                        execution_status="completed",
                        query=result.query,
                        capability_label=tool_def.label,
                        items=items,
                    )
                )
            except Exception as exc:
                out.append(
                    ExecutedStep(
                        id=step.id,
                        action=tool_def.name,
                        tool_name=tool_def.name,
                        success=False,
                        reason=step.reason,
                        execution_status="failed",
                        query=user_text,
                        capability_label=tool_def.label,
                        items=[],
                        error=str(exc),
                    )
                )
        return out
