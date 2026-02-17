"""Executive controller: decides what the agent does next."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from covenant.core.working_memory import AgentState, WorkingMemory

if TYPE_CHECKING:
    from covenant.llm.client import LLMClient

log = logging.getLogger(__name__)


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    if not text:
        raise ValueError("Empty LLM response")

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code blocks: ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1).strip())

    # Try finding first { ... } in the text
    start = text.find("{")
    if start != -1:
        # Find the matching closing brace
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])

    raise ValueError(f"Could not extract JSON from LLM response: {text[:200]}")


class Action(str, Enum):
    RESPOND = "respond"
    ASK = "ask"
    TOOL = "tool"
    RETRIEVE_MORE = "retrieve_more"
    REFLECT = "reflect"
    STOP = "stop"


@dataclass
class Decision:
    action: Action
    payload: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


class ExecutiveController:
    """Decides the agent's next action. Uses LLM when available, falls back to rules."""

    def __init__(self, llm_client: LLMClient | None = None, tool_list: str = "None"):
        self.llm_client = llm_client
        self.tool_list = tool_list

    async def decide(self, wm: WorkingMemory) -> Decision:
        if self.llm_client is not None:
            try:
                return await self._llm_decide(wm)
            except Exception:
                log.warning("LLM decide failed, falling back to LLM chat", exc_info=True)
            # Fallback: use the LLM for a direct conversational response
            try:
                return await self._llm_chat_fallback(wm)
            except Exception:
                log.warning("LLM chat fallback also failed, falling back to rules", exc_info=True)
        return self._rule_decide(wm)

    async def _llm_decide(self, wm: WorkingMemory) -> Decision:
        from covenant.llm.prompts import format_executive_prompt

        messages = format_executive_prompt(wm.snapshot(), self.tool_list)
        resp = await self.llm_client.complete(messages)

        parsed = _extract_json(resp.text)
        action = Action(parsed["action"])
        return Decision(
            action=action,
            payload=parsed.get("payload", {}),
            reasoning=parsed.get("reasoning", ""),
        )

    async def _llm_chat_fallback(self, wm: WorkingMemory) -> Decision:
        """Generate a direct conversational response when executive decision parsing fails."""
        # Extract user message from observations
        user_msg = ""
        for obs in reversed(wm.recent_observations):
            if obs.startswith("user: "):
                user_msg = obs[6:]
                break

        messages = [
            {"role": "system", "content": "You are Covenant, a helpful AI assistant. Respond concisely."},
            {"role": "user", "content": user_msg or wm.goal or "Hello"},
        ]
        resp = await self.llm_client.complete(messages)
        return Decision(
            action=Action.RESPOND,
            payload={"text": resp.text},
            reasoning="direct LLM chat fallback",
        )

    def _rule_decide(self, wm: WorkingMemory) -> Decision:
        if wm.budget <= 0:
            return Decision(action=Action.STOP, reasoning="budget exhausted")

        if wm.state == AgentState.IDLE:
            if wm.goal:
                return Decision(
                    action=Action.RESPOND,
                    payload={"text": f"Starting work on: {wm.goal}"},
                    reasoning="new goal received",
                )
            return Decision(action=Action.STOP, reasoning="no goal set")

        if wm.state == AgentState.THINKING:
            if not wm.recent_observations:
                return Decision(action=Action.RETRIEVE_MORE, reasoning="need more context")
            return Decision(
                action=Action.RESPOND,
                payload={"text": f"Based on observations, working on: {wm.goal}"},
                reasoning="have observations to work with",
            )

        if wm.state == AgentState.ACTING:
            return Decision(action=Action.REFLECT, reasoning="action completed, should reflect")

        if wm.state == AgentState.REFLECTING:
            return Decision(
                action=Action.RESPOND,
                payload={"text": "Reflection complete."},
                reasoning="reflection done",
            )

        return Decision(action=Action.STOP, reasoning="unknown state")
