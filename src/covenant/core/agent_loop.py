"""Agent loop: the main control flow of the Covenant agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from covenant.core.executive import Action, ExecutiveController
from covenant.core.reflection import micro_reflect
from covenant.core.safety import SafetyGate
from covenant.core.working_memory import AgentState, WorkingMemory


@dataclass
class StepResult:
    action: Action
    payload: dict[str, Any] = field(default_factory=dict)
    blocked: bool = False
    block_reason: str = ""


class AgentLoop:
    def __init__(
        self,
        executive: ExecutiveController | None = None,
        safety: SafetyGate | None = None,
        llm_client: Any = None,
        retriever: Any = None,
        tool_router: Any = None,
    ):
        self.llm_client = llm_client
        self.retriever = retriever
        self.tool_router = tool_router
        self.executive = executive or ExecutiveController(llm_client=llm_client)
        self.safety = safety or SafetyGate()
        self.wm = WorkingMemory()
        self.history: list[StepResult] = []

    async def step(self, user_input: str | None = None) -> StepResult:
        """Execute one iteration: Ingest -> Update WM -> Retrieve -> Decide -> Safety -> Execute -> Reflect."""
        # 1. Ingest
        if user_input:
            self._ingest(user_input)

        # 2. Update WM state
        self.wm.state = AgentState.THINKING

        # 3. Retrieve (placeholder - wired in Phase 4)
        await self._retrieve()

        # 4. Decide
        decision = await self.executive.decide(self.wm)

        # 5. Safety check
        safety_result = self.safety.check(decision)
        if not safety_result.allowed:
            result = StepResult(
                action=decision.action,
                payload=decision.payload,
                blocked=True,
                block_reason=safety_result.reason,
            )
            self.history.append(result)
            return result

        # 6. Execute
        output = await self._execute(decision)

        # 7. Write memory (placeholder - stores episode in Phase 1 DB)

        # 8. Reflect
        observation = f"action={decision.action.value} output={output}"
        self.wm = micro_reflect(self.wm, observation)

        result = StepResult(action=decision.action, payload=decision.payload)
        self.history.append(result)
        return result

    async def run(self, goal: str, max_steps: int = 10) -> list[StepResult]:
        """Run the agent loop until STOP or max_steps reached."""
        self.wm.goal = goal
        self.wm.budget = max_steps
        results = []

        # First step with goal as input
        result = await self.step(goal)
        results.append(result)

        # Continue until stop or budget exhausted
        for _ in range(max_steps - 1):
            if result.action == Action.STOP:
                break
            result = await self.step()
            results.append(result)

        return results

    def _ingest(self, user_input: str) -> None:
        if not self.wm.goal:
            self.wm.goal = user_input
        self.wm.add_observation(f"user: {user_input}")

    async def _retrieve(self) -> None:
        if self.retriever is not None:
            results = await self.retriever.retrieve(self.wm.goal, limit=5)
            for r in results:
                self.wm.add_observation(f"memory: {r}")

    async def _execute(self, decision) -> str:
        if decision.action == Action.TOOL and self.tool_router is not None:
            tool_name = decision.payload.get("tool", "")
            tool_args = decision.payload.get("args", {})
            return await self.tool_router.execute(tool_name, **tool_args)

        if decision.action == Action.RESPOND:
            return decision.payload.get("text", "")

        return f"executed {decision.action.value}"
