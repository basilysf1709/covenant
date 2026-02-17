"""Working memory: the agent's scratchpad during a task."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, model_validator


class AgentState(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    REFLECTING = "reflecting"


class WorkingMemory(BaseModel):
    goal: str = ""
    plan: list[str] = []
    state: AgentState = AgentState.IDLE
    recent_observations: list[str] = []
    open_loops: list[str] = []
    budget: int = 10
    context: dict[str, Any] = {}

    MAX_SIZE_BYTES: int = 3072  # 3KB

    @model_validator(mode="after")
    def enforce_size_limit(self) -> WorkingMemory:
        if self._byte_size() > self.MAX_SIZE_BYTES:
            self.compact()
        return self

    def _byte_size(self) -> int:
        return len(self.model_dump_json().encode())

    def compact(self) -> None:
        """Trim working memory to fit within size limit."""
        # Drop oldest observations first
        while self._byte_size() > self.MAX_SIZE_BYTES and len(self.recent_observations) > 1:
            self.recent_observations.pop(0)
        # Then trim context
        while self._byte_size() > self.MAX_SIZE_BYTES and self.context:
            oldest_key = next(iter(self.context))
            del self.context[oldest_key]
        # Then trim plan
        while self._byte_size() > self.MAX_SIZE_BYTES and len(self.plan) > 1:
            self.plan.pop(0)

    def add_observation(self, obs: str) -> None:
        self.recent_observations.append(obs)
        if self._byte_size() > self.MAX_SIZE_BYTES:
            self.compact()

    def snapshot(self) -> str:
        """JSON snapshot for injecting into prompts."""
        return json.dumps(self.model_dump(exclude={"MAX_SIZE_BYTES"}), indent=2)
