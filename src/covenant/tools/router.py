"""Tool registry and dispatch system."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, str] = field(default_factory=dict)
    fn: Callable | None = None


class ToolRouter:
    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    def register(self, name: str, description: str, parameters: dict[str, str] | None = None):
        """Decorator to register a tool function."""
        def decorator(fn: Callable) -> Callable:
            self._tools[name] = ToolSpec(
                name=name,
                description=description,
                parameters=parameters or {},
                fn=fn,
            )
            return fn
        return decorator

    def add_tool(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def list_tools(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def list_tools_text(self) -> str:
        if not self._tools:
            return "None"
        lines = []
        for t in self._tools.values():
            params = ", ".join(f"{k}: {v}" for k, v in t.parameters.items())
            lines.append(f"- {t.name}({params}): {t.description}")
        return "\n".join(lines)

    async def execute(self, name: str, **kwargs: Any) -> str:
        spec = self._tools.get(name)
        if spec is None:
            return f"Error: unknown tool '{name}'"
        if spec.fn is None:
            return f"Error: tool '{name}' has no implementation"
        try:
            if inspect.iscoroutinefunction(spec.fn):
                result = await spec.fn(**kwargs)
            else:
                result = spec.fn(**kwargs)
            return str(result)
        except Exception as e:
            return f"Error: {e}"
