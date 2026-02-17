"""Eval harness: run scenarios and generate reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from covenant.core.agent_loop import AgentLoop
from covenant.eval.scenarios import Scenario


@dataclass
class ScenarioResult:
    scenario: Scenario
    passed: bool
    steps: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""


@dataclass
class EvalReport:
    results: list[ScenarioResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    def summary(self) -> str:
        lines = [f"Eval Report: {self.passed}/{self.total} passed\n"]
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"  [{status}] {r.scenario.name}: {r.reason}")
        return "\n".join(lines)


class EvalHarness:
    def __init__(self, agent_factory=None):
        self.agent_factory = agent_factory or (lambda: AgentLoop())

    async def run_scenario(self, scenario: Scenario) -> ScenarioResult:
        agent = self.agent_factory()
        results = await agent.run(scenario.goal, max_steps=scenario.max_steps)

        steps = [
            {"action": r.action.value, "payload": r.payload, "blocked": r.blocked}
            for r in results
        ]

        # Check if any expected action was performed
        actions_taken = {r.action.value for r in results}
        expected_met = any(ea in actions_taken for ea in scenario.expected_actions)

        if expected_met:
            return ScenarioResult(
                scenario=scenario,
                passed=True,
                steps=steps,
                reason=f"Expected actions {scenario.expected_actions} found in {actions_taken}",
            )
        else:
            return ScenarioResult(
                scenario=scenario,
                passed=False,
                steps=steps,
                reason=f"Expected {scenario.expected_actions}, got {actions_taken}",
            )

    async def run_all(self, scenarios: list[Scenario]) -> EvalReport:
        report = EvalReport()
        for scenario in scenarios:
            result = await self.run_scenario(scenario)
            report.results.append(result)
        return report

    def report(self, eval_report: EvalReport) -> str:
        return eval_report.summary()
