"""Tests for the agent loop."""

import json

from sqlalchemy import select

from covenant.core.agent_loop import AgentLoop
from covenant.core.executive import Action
from covenant.memory.models import Episode
from covenant.memory.retrieval import MemoryRetriever


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


# --- Memory integration tests ---


async def test_step_writes_episode(db_session):
    """After one step, an Episode row should exist in the DB."""
    agent = AgentLoop(session=db_session)
    await agent.step("remember the sky is blue")

    result = await db_session.execute(select(Episode))
    episodes = result.scalars().all()
    assert len(episodes) == 1
    assert "remember the sky is blue" in episodes[0].goal


async def test_step_without_session_no_episode():
    """Volatile mode (no session) should not error and writes nothing."""
    agent = AgentLoop()  # no session
    result = await agent.step("hello volatile")
    assert result.action in Action


async def test_run_writes_multiple_episodes(db_session):
    """Episode count should match the number of steps executed."""
    agent = AgentLoop(session=db_session)
    results = await agent.run("do stuff", max_steps=3)

    result = await db_session.execute(select(Episode))
    episodes = result.scalars().all()
    assert len(episodes) == len(results)


async def test_retrieval_sees_stored_episodes(db_session):
    """Step 2 should be able to retrieve the episode written in step 1."""
    agent = AgentLoop(session=db_session)

    # Step 1: write an episode about "deadline is Friday"
    await agent.step("remember that the project deadline is Friday")

    # Verify it was written
    result = await db_session.execute(select(Episode))
    episodes = result.scalars().all()
    assert len(episodes) >= 1

    # Step 2: the retriever should find the stored episode
    retriever = MemoryRetriever(db_session)
    results = await retriever.retrieve("deadline", limit=5)
    assert len(results) >= 1
    assert any("deadline" in r.content.lower() or "friday" in r.content.lower() for r in results)


async def test_fake_llm_with_db_session(db_session):
    """FakeLLMClient + DB session should work together."""
    from covenant.llm.client import FakeLLMClient

    responses = [
        json.dumps({"action": "respond", "payload": {"text": "Hello!"}, "reasoning": "greeting"}),
        json.dumps({"action": "stop", "payload": {}, "reasoning": "done"}),
    ]
    fake = FakeLLMClient(responses=responses)
    agent = AgentLoop(llm_client=fake, session=db_session)
    results = await agent.run("greet me", max_steps=3)

    assert any(r.action == Action.RESPOND for r in results)

    result = await db_session.execute(select(Episode))
    episodes = result.scalars().all()
    assert len(episodes) >= 1


async def test_auto_wired_retriever(db_session):
    """When session is provided and no retriever, MemoryRetriever should be auto-created."""
    agent = AgentLoop(session=db_session)
    assert agent.retriever is not None
    assert isinstance(agent.retriever, MemoryRetriever)


async def test_explicit_retriever_not_overridden(db_session):
    """A custom retriever should not be replaced by auto-wiring."""

    class CustomRetriever:
        async def retrieve(self, query, limit=10):
            return []

    custom = CustomRetriever()
    agent = AgentLoop(session=db_session, retriever=custom)
    assert agent.retriever is custom


async def test_embed_fn_stores_embeddings(db_session):
    """When embed_fn is provided, episodes should be stored with embeddings."""

    async def fake_embed(text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    agent = AgentLoop(session=db_session, embed_fn=fake_embed)
    await agent.step("remember the sky is blue")

    result = await db_session.execute(select(Episode))
    episodes = result.scalars().all()
    assert len(episodes) == 1
    assert episodes[0].embedding is not None
    import json
    stored = json.loads(episodes[0].embedding)
    assert stored == [0.1, 0.2, 0.3]


async def test_embed_fn_wired_to_retriever(db_session):
    """embed_fn should be passed through to the auto-wired MemoryRetriever."""

    async def fake_embed(text: str) -> list[float]:
        return [0.0] * 4

    agent = AgentLoop(session=db_session, embed_fn=fake_embed)
    assert agent.retriever is not None
    assert agent.retriever.embed_fn is fake_embed
