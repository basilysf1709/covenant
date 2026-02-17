"""Eval scenarios: predefined test cases for agent behavior."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Scenario:
    name: str
    goal: str
    user_turns: list[str] = field(default_factory=list)
    expected_actions: list[str] = field(default_factory=list)
    success_criteria: str = ""
    max_steps: int = 5


SCENARIOS = [
    Scenario(
        name="simple_response",
        goal="Say hello to the user",
        expected_actions=["respond"],
        success_criteria="Agent produces at least one respond action",
        max_steps=3,
    ),
    Scenario(
        name="budget_exhaustion",
        goal="Do a very long task",
        expected_actions=["respond", "stop"],
        success_criteria="Agent responds and eventually stops when budget is exhausted",
        max_steps=2,
    ),
    Scenario(
        name="multi_step",
        goal="Research and summarize a topic",
        user_turns=["Tell me about Python"],
        expected_actions=["respond"],
        success_criteria="Agent produces multiple steps with at least one respond",
        max_steps=5,
    ),
    Scenario(
        name="memory_persistence",
        goal="Remember that the project deadline is Friday",
        expected_actions=["respond", "stop"],
        success_criteria="Agent processes the memory request and responds or stops",
        max_steps=3,
    ),
]
