from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .llm_client import OpenAICompatibleClient, extract_first_json_object


@dataclass(frozen=True)
class PlannedStep:
    id: str
    tool: str
    args: dict[str, Any]
    reason: str


@dataclass(frozen=True)
class PlanResult:
    mode: str
    reason: str
    confidence: float
    steps: list[PlannedStep]
    planner_raw: str


class LlmPlanner:
    def __init__(
        self,
        *,
        llm: OpenAICompatibleClient,
        max_steps: int,
    ) -> None:
        self._llm = llm
        self._max_steps = max(1, min(8, int(max_steps)))

    def plan(
        self,
        *,
        user_text: str,
        memory_context: list[dict[str, str]],
        tool_registry_prompt: str,
    ) -> PlanResult:
        model_output = self._llm.complete(
            messages=[
                {"role": "system", "content": self._build_system_prompt(tool_registry_prompt)},
                {
                    "role": "user",
                    "content": self._build_user_prompt(
                        user_text=user_text,
                        memory_context=memory_context,
                        max_steps=self._max_steps,
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=900,
        )
        parsed = extract_first_json_object(model_output)
        if isinstance(parsed, dict):
            return self._validate_plan(parsed, raw=model_output)

        repaired = self._llm.complete(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Convert the input into valid planner JSON only.\n"
                        "No markdown. No explanation. Output one JSON object.\n"
                        "Required keys: mode, reason, confidence, steps."
                    ),
                },
                {"role": "user", "content": model_output},
            ],
            temperature=0.0,
            max_tokens=700,
        )
        repaired_parsed = extract_first_json_object(repaired)
        if not isinstance(repaired_parsed, dict):
            raise RuntimeError("Planner did not return valid JSON.")
        return self._validate_plan(repaired_parsed, raw=repaired)

    def _validate_plan(self, payload: dict[str, object], *, raw: str) -> PlanResult:
        mode = str(payload.get("mode") or "").strip().lower()
        if mode not in {"direct_response", "tool_pipeline"}:
            raise RuntimeError("Planner JSON must set mode to direct_response or tool_pipeline.")

        reason = str(payload.get("reason") or "").strip() or "Planner decision generated."
        confidence_raw = payload.get("confidence")
        confidence = 0.5
        if isinstance(confidence_raw, (int, float)):
            confidence = max(0.0, min(1.0, float(confidence_raw)))

        raw_steps = payload.get("steps")
        steps: list[PlannedStep] = []
        if isinstance(raw_steps, list):
            for idx, row in enumerate(raw_steps[: self._max_steps], start=1):
                if not isinstance(row, dict):
                    continue
                tool = str(row.get("tool") or "").strip()
                if not tool:
                    continue
                args = row.get("args")
                normalized_args = args if isinstance(args, dict) else {}
                step_reason = str(row.get("reason") or "").strip() or f"Step {idx}"
                step_id = str(row.get("id") or "").strip() or f"step_{idx}"
                steps.append(
                    PlannedStep(
                        id=step_id,
                        tool=tool,
                        args=normalized_args,
                        reason=step_reason,
                    )
                )

        if mode == "direct_response" and steps:
            steps = []
        if mode == "tool_pipeline" and not steps:
            raise RuntimeError("Planner selected tool_pipeline but returned no valid steps.")

        return PlanResult(
            mode=mode,
            reason=reason,
            confidence=confidence,
            steps=steps,
            planner_raw=raw,
        )

    @staticmethod
    def _build_system_prompt(tool_registry_prompt: str) -> str:
        return (
            "You are the Cortex Planner. Decide intent semantically from full context.\n"
            "Never use lexical or keyword matching rules. No regex logic.\n"
            "Choose exactly one mode:\n"
            "1) direct_response: no tools required.\n"
            "2) tool_pipeline: one or more tool calls required.\n\n"
            "Output strict JSON only.\n"
            "JSON schema:\n"
            "{\n"
            '  "mode": "direct_response" | "tool_pipeline",\n'
            '  "reason": "short decision rationale",\n'
            '  "confidence": 0.0-1.0,\n'
            '  "steps": [\n'
            "    {\n"
            '      "id": "step_1",\n'
            '      "tool": "<registered_tool_name>",\n'
            '      "reason": "why this step is needed",\n'
            '      "args": { ... validated tool args ... }\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "If mode=direct_response then steps must be an empty list.\n"
            "If mode=tool_pipeline then steps must be non-empty and ordered.\n\n"
            "Tool intent guidance:\n"
            "- 'check my email', 'new emails', 'inbox' => use google_gmail read operation.\n"
            "- 'check calendar', 'events this week' => use google_calendar read operation.\n"
            "- 'find file in drive' => use google_drive search operation.\n"
            "- If operation is omitted, executor defaults to safe read behavior.\n\n"
            f"{tool_registry_prompt}\n"
        )

    @staticmethod
    def _build_user_prompt(
        *,
        user_text: str,
        memory_context: list[dict[str, str]],
        max_steps: int,
    ) -> str:
        context_lines: list[str] = []
        for row in memory_context:
            role = str(row.get("role") or "").strip().lower()
            content = str(row.get("content") or "").strip()
            if role and content:
                context_lines.append(f"{role.upper()}: {content}")
        context_blob = "\n".join(context_lines[-20:])
        return (
            f"MAX_STEPS: {max_steps}\n"
            f"LATEST_USER_MESSAGE:\n{user_text.strip()}\n\n"
            f"CONVERSATION_CONTEXT:\n{context_blob}\n"
        )
