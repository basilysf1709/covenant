"""Executive controller: decides what the agent does next."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from covenant.core.working_memory import AgentState, WorkingMemory

if TYPE_CHECKING:
    from covenant.llm.client import LLMClient

log = logging.getLogger(__name__)


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
                log.warning("LLM decide failed, falling back to rules", exc_info=True)
        return self._rule_decide(wm)

    async def _llm_decide(self, wm: WorkingMemory) -> Decision:
        from covenant.llm.prompts import format_executive_prompt

        messages = format_executive_prompt(wm.snapshot(), self.tool_list)
        resp = await self.llm_client.complete(messages)

        parsed = json.loads(resp.text)
        action = Action(parsed["action"])
        return Decision(
            action=action,
            payload=parsed.get("payload", {}),
            reasoning=parsed.get("reasoning", ""),
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
