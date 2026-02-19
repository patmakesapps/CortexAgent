from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cortexagent.models import AgentDecision, AgentChatResponse
from cortexagent.services.cortexltm_client import CortexLtmClient
from cortexagent.services.executor import ExecutedStep, ToolExecutor
from cortexagent.services.llm_client import OpenAICompatibleClient
from cortexagent.services.planner import LlmPlanner


@dataclass(frozen=True)
class AgentRuntimeResult:
    response: AgentChatResponse
    trace: dict[str, object]


class AgentOrchestrator:
    _MAX_EVENT_CONTENT_CHARS = 6000

    def __init__(
        self,
        *,
        planner: LlmPlanner,
        executor: ToolExecutor,
        cortexltm: CortexLtmClient,
        synthesis_llm: OpenAICompatibleClient | None,
        planner_context_messages: int,
    ) -> None:
        self._planner = planner
        self._executor = executor
        self._cortexltm = cortexltm
        self._synthesis_llm = synthesis_llm
        self._planner_context_messages = max(4, int(planner_context_messages))

    def chat(
        self,
        *,
        thread_id: str,
        user_text: str,
        short_term_limit: int | None,
        authorization: str | None,
        planner_tool_registry_prompt: str,
        tool_meta: dict[str, object],
    ) -> AgentRuntimeResult:
        memory_context = self._cortexltm.build_memory_context(
            thread_id=thread_id,
            latest_user_text=user_text,
            short_term_limit=short_term_limit,
            authorization=authorization,
        )
        planner_result = self._planner.plan(
            user_text=user_text,
            memory_context=memory_context[-self._planner_context_messages :],
            tool_registry_prompt=planner_tool_registry_prompt,
        )

        if planner_result.mode == "direct_response":
            assistant_text = self._cortexltm.chat(
                thread_id=thread_id,
                text=user_text,
                short_term_limit=short_term_limit,
                authorization=authorization,
            )
            decision = AgentDecision(
                action="chat",
                reason=planner_result.reason,
                confidence=planner_result.confidence,
            )
            response = AgentChatResponse(
                thread_id=thread_id,
                response=assistant_text,
                decision=decision,
                sources=[],
                tool_pipeline=[],
            )
            return AgentRuntimeResult(
                response=response,
                trace={
                    "planner_mode": planner_result.mode,
                    "planner_reason": planner_result.reason,
                    "planner_confidence": planner_result.confidence,
                    "planner_raw": planner_result.planner_raw,
                },
            )

        self._cortexltm.create_event(
            thread_id=thread_id,
            actor="user",
            content=user_text,
            meta={"source": "agent_user"},
            authorization=authorization,
        )
        executed = self._executor.execute_steps(
            thread_id=thread_id,
            user_text=user_text,
            plan_steps=planner_result.steps,
            tool_meta=tool_meta,
        )
        assistant_text = self._synthesize_tool_reply(
            executed_steps=executed,
        )
        self._cortexltm.create_event(
            thread_id=thread_id,
            actor="assistant",
            content=self._sanitize_event_content(assistant_text),
            meta={"source": "agent_tool_pipeline"},
            authorization=authorization,
        )

        decision = AgentDecision(
            action="orchestration" if len(executed) > 1 else executed[0].action,
            reason=planner_result.reason,
            confidence=planner_result.confidence,
        )
        response = AgentChatResponse(
            thread_id=thread_id,
            response=assistant_text,
            decision=decision,
            sources=self._collect_sources(executed),
            tool_pipeline=[self._step_payload(step) for step in executed],
        )
        return AgentRuntimeResult(
            response=response,
            trace={
                "planner_mode": planner_result.mode,
                "planner_reason": planner_result.reason,
                "planner_confidence": planner_result.confidence,
                "planner_raw": planner_result.planner_raw,
                "executed_steps": [self._step_payload(step) for step in executed],
            },
        )

    def _synthesize_tool_reply(
        self,
        *,
        executed_steps: list[ExecutedStep],
    ) -> str:
        # Deterministic tool-grounded rendering to prevent context leakage/hallucinated blending.
        return self._fallback_tool_summary(executed_steps)

    @staticmethod
    def _fallback_tool_summary(executed_steps: list[ExecutedStep]) -> str:
        successful = [step for step in executed_steps if step.success]
        failed = [step for step in executed_steps if not step.success]
        if not successful:
            lines = ["I couldn't complete the requested tool actions."]
            for step in failed[:3]:
                lines.append(f"- {step.capability_label}: {step.error or 'failed'}")
            lines.append("Please retry with a narrower request.")
            return "\n".join(lines)
        lines: list[str] = []
        for step in successful:
            lines.append(f"{step.capability_label}:")
            if not step.items:
                lines.append("- Completed with no returned items.")
                continue
            for item in step.items[:5]:
                title = item.get("title", "").strip()
                snippet = item.get("snippet", "").strip()
                if len(snippet) > 220:
                    snippet = snippet[:220].rstrip() + "..."
                url = item.get("url", "").strip()
                row = f"- {title or 'Result'}"
                if snippet:
                    row += f" | {snippet}"
                if url:
                    row += f" | {url}"
                lines.append(row)
        if failed:
            lines.append("")
            lines.append("Some steps failed:")
            for step in failed[:3]:
                lines.append(f"- {step.capability_label}: {step.error or 'failed'}")
        rendered = "\n".join(lines).strip()
        return AgentOrchestrator._sanitize_event_content(rendered)

    @staticmethod
    def _sanitize_event_content(value: str) -> str:
        text = (value or "").strip()
        if not text:
            return "Tool execution completed."
        if len(text) <= AgentOrchestrator._MAX_EVENT_CONTENT_CHARS:
            return text
        return text[: AgentOrchestrator._MAX_EVENT_CONTENT_CHARS - 3].rstrip() + "..."

    @staticmethod
    def _collect_sources(executed_steps: list[ExecutedStep]) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        seen: set[str] = set()
        for step in executed_steps:
            for item in step.items:
                url = item.get("url", "").strip()
                title = item.get("title", "").strip()
                if not url or url in seen:
                    continue
                seen.add(url)
                out.append({"title": title or url, "url": url})
        return out

    @staticmethod
    def _step_payload(step: ExecutedStep) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": step.id,
            "action": step.action,
            "tool_name": step.tool_name,
            "success": step.success,
            "reason": step.reason,
            "execution_status": step.execution_status,
            "query": step.query,
            "capability_label": step.capability_label,
            "items": step.items,
        }
        if step.error:
            payload["error"] = step.error
        return payload
