"""Safety gate: checks actions before execution."""

from __future__ import annotations

from dataclasses import dataclass

from covenant.core.executive import Action, Decision


BLOCKED_TOOLS = frozenset({"shell", "exec", "rm", "delete_all"})


@dataclass
class SafetyResult:
    allowed: bool
    reason: str = ""


class SafetyGate:
    """Permissive stub that blocks specific dangerous tools."""

    def check(self, decision: Decision) -> SafetyResult:
        if decision.action == Action.TOOL:
            tool_name = decision.payload.get("tool", "")
            if tool_name in BLOCKED_TOOLS:
                return SafetyResult(allowed=False, reason=f"tool '{tool_name}' is blocked")
        return SafetyResult(allowed=True)
