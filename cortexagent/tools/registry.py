from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import Tool


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    label: str
    description: str
    tool: Tool
    schema: dict[str, object]

    def validate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(args, dict):
            raise ValueError(f"Tool '{self.name}' args must be an object.")

        required = self.schema.get("required")
        required_fields = required if isinstance(required, list) else []
        for field in required_fields:
            if not isinstance(field, str):
                continue
            if field not in args:
                raise ValueError(f"Tool '{self.name}' missing required arg '{field}'.")

        properties = self.schema.get("properties")
        if not isinstance(properties, dict):
            return args

        clean: dict[str, Any] = {}
        for key, value in args.items():
            prop = properties.get(key)
            if not isinstance(prop, dict):
                continue
            expected = prop.get("type")
            if expected == "string":
                if not isinstance(value, str):
                    raise ValueError(f"Tool '{self.name}' arg '{key}' must be a string.")
                clean[key] = value
            elif expected == "integer":
                if not isinstance(value, int):
                    raise ValueError(f"Tool '{self.name}' arg '{key}' must be an integer.")
                clean[key] = value
            elif expected == "object":
                if not isinstance(value, dict):
                    raise ValueError(f"Tool '{self.name}' arg '{key}' must be an object.")
                clean[key] = value
            elif expected == "array":
                if not isinstance(value, list):
                    raise ValueError(f"Tool '{self.name}' arg '{key}' must be an array.")
                clean[key] = value
            else:
                clean[key] = value

        for field in required_fields:
            if isinstance(field, str) and field in args and field not in clean:
                clean[field] = args[field]
        return clean


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        *,
        tool: Tool,
        label: str,
        description: str,
        schema: dict[str, object],
    ) -> None:
        self._tools[tool.name] = ToolDefinition(
            name=tool.name,
            label=label,
            description=description,
            tool=tool,
            schema=schema,
        )

    def get_definition(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ValueError(f"Tool '{name}' is not registered.") from exc

    def list_tools(self) -> list[str]:
        return sorted(self._tools.keys())

    def render_for_prompt(self) -> str:
        lines = ["TOOL REGISTRY"]
        for name in self.list_tools():
            tool = self._tools[name]
            lines.append(f"- {tool.name}: {tool.description}")
            properties = tool.schema.get("properties")
            if isinstance(properties, dict):
                for arg_name, arg_spec in properties.items():
                    if not isinstance(arg_name, str) or not isinstance(arg_spec, dict):
                        continue
                    arg_type = str(arg_spec.get("type") or "any")
                    arg_desc = str(arg_spec.get("description") or "").strip()
                    lines.append(f"  - {arg_name} ({arg_type}): {arg_desc}")
        return "\n".join(lines)
