"""Tests for the agent loop."""

from covenant.core.agent_loop import AgentLoop
from covenant.core.executive import Action


async def test_single_step():
    agent = AgentLoop()
    result = await agent.step("hello")
    assert result.action in Action


async def test_run_completes():
    agent = AgentLoop()
    results = await agent.run("test goal", max_steps=5)
    assert len(results) >= 1
    assert results[-1].action == Action.STOP or len(results) == 5


async def test_safety_blocks_tool():
    agent = AgentLoop()
    agent.wm.goal = "test"
    # Manually force a tool decision that should be blocked
    from covenant.core.executive import Decision
    from covenant.core.safety import SafetyGate

    gate = SafetyGate()
    decision = Decision(action=Action.TOOL, payload={"tool": "shell"})
    result = gate.check(decision)
    assert not result.allowed


async def test_budget_depletes():
    agent = AgentLoop()
    results = await agent.run("goal", max_steps=3)
    # Budget should decrease with each step
    assert agent.wm.budget < 3
