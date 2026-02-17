"""Tests for the executive controller."""

from covenant.core.executive import Action, ExecutiveController
from covenant.core.working_memory import AgentState, WorkingMemory


async def test_idle_with_goal():
    ec = ExecutiveController()
    wm = WorkingMemory(goal="do something", state=AgentState.IDLE)
    decision = await ec.decide(wm)
    assert decision.action == Action.RESPOND


async def test_idle_no_goal():
    ec = ExecutiveController()
    wm = WorkingMemory(state=AgentState.IDLE)
    decision = await ec.decide(wm)
    assert decision.action == Action.STOP


async def test_budget_exhausted():
    ec = ExecutiveController()
    wm = WorkingMemory(goal="do something", budget=0)
    decision = await ec.decide(wm)
    assert decision.action == Action.STOP


async def test_thinking_no_observations():
    ec = ExecutiveController()
    wm = WorkingMemory(goal="goal", state=AgentState.THINKING)
    decision = await ec.decide(wm)
    assert decision.action == Action.RETRIEVE_MORE


async def test_thinking_with_observations():
    ec = ExecutiveController()
    wm = WorkingMemory(goal="goal", state=AgentState.THINKING, recent_observations=["saw it"])
    decision = await ec.decide(wm)
    assert decision.action == Action.RESPOND


async def test_acting_triggers_reflect():
    ec = ExecutiveController()
    wm = WorkingMemory(state=AgentState.ACTING)
    decision = await ec.decide(wm)
    assert decision.action == Action.REFLECT
