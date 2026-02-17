"""Tests for LLM client and executive LLM integration."""

import json

from covenant.core.agent_loop import AgentLoop
from covenant.core.executive import Action, ExecutiveController
from covenant.core.working_memory import AgentState, WorkingMemory
from covenant.llm.client import FakeLLMClient
from covenant.llm.prompts import format_executive_prompt


async def test_fake_llm_client():
    client = FakeLLMClient(responses=["hello", "world"])
    resp1 = await client.complete([{"role": "user", "content": "hi"}])
    assert resp1.text == "hello"
    resp2 = await client.complete([{"role": "user", "content": "yo"}])
    assert resp2.text == "world"
    assert len(client.call_log) == 2


async def test_fake_llm_embed():
    client = FakeLLMClient()
    emb = await client.embed("test")
    assert len(emb) == 1536


def test_format_executive_prompt():
    wm = WorkingMemory(goal="test")
    messages = format_executive_prompt(wm.snapshot())
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "respond" in messages[0]["content"]


async def test_executive_with_llm():
    response = json.dumps({"action": "respond", "payload": {"text": "I will help"}, "reasoning": "test"})
    client = FakeLLMClient(responses=[response])
    ec = ExecutiveController(llm_client=client)
    wm = WorkingMemory(goal="do something", state=AgentState.THINKING)
    decision = await ec.decide(wm)
    assert decision.action == Action.RESPOND
    assert decision.payload["text"] == "I will help"


async def test_executive_llm_fallback():
    """When LLM returns invalid JSON, falls back to rules."""
    client = FakeLLMClient(responses=["not json at all"])
    ec = ExecutiveController(llm_client=client)
    wm = WorkingMemory(goal="do something", state=AgentState.THINKING, recent_observations=["obs"])
    decision = await ec.decide(wm)
    # Should fall back to rule-based decision
    assert decision.action == Action.RESPOND


async def test_agent_loop_with_fake_llm():
    responses = [
        json.dumps({"action": "respond", "payload": {"text": "Working on it"}, "reasoning": "start"}),
        json.dumps({"action": "stop", "payload": {}, "reasoning": "done"}),
    ]
    client = FakeLLMClient(responses=responses)
    agent = AgentLoop(llm_client=client)
    results = await agent.run("test", max_steps=5)
    assert any(r.action == Action.RESPOND for r in results)
