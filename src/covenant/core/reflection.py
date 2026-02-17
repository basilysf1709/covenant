"""Micro-reflection: quick post-action introspection."""

from __future__ import annotations

from covenant.core.working_memory import WorkingMemory


def micro_reflect(wm: WorkingMemory, observation: str) -> WorkingMemory:
    """Append observation, decrement budget, compact if needed."""
    wm.add_observation(observation)
    wm.budget = max(0, wm.budget - 1)
    return wm
