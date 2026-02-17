"""Tests for the eval harness."""

from covenant.eval.harness import EvalHarness, EvalReport
from covenant.eval.scenarios import SCENARIOS, Scenario


async def test_run_single_scenario():
    scenario = Scenario(
        name="test_respond",
        goal="Hello",
        expected_actions=["respond"],
        max_steps=3,
    )
    harness = EvalHarness()
    result = await harness.run_scenario(scenario)
    assert result.passed
    assert len(result.steps) >= 1


async def test_run_all_scenarios():
    harness = EvalHarness()
    report = await harness.run_all(SCENARIOS)
    assert report.total == len(SCENARIOS)
    assert report.passed + report.failed == report.total


async def test_report_generation():
    harness = EvalHarness()
    report = await harness.run_all(SCENARIOS)
    summary = harness.report(report)
    assert "Eval Report" in summary
    assert "PASS" in summary or "FAIL" in summary


async def test_scenario_with_expected_stop():
    scenario = Scenario(
        name="test_stop",
        goal="Do something",
        expected_actions=["stop"],
        max_steps=2,
    )
    harness = EvalHarness()
    result = await harness.run_scenario(scenario)
    # The agent should eventually stop
    assert len(result.steps) >= 1


async def test_eval_report_properties():
    report = EvalReport()
    assert report.total == 0
    assert report.passed == 0
    assert report.failed == 0


async def test_custom_agent_factory():
    from covenant.core.agent_loop import AgentLoop

    factory_called = False

    def custom_factory():
        nonlocal factory_called
        factory_called = True
        return AgentLoop()

    harness = EvalHarness(agent_factory=custom_factory)
    scenario = Scenario(name="test", goal="hi", expected_actions=["respond"], max_steps=2)
    await harness.run_scenario(scenario)
    assert factory_called
