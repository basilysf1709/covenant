"""Tests for MemoryRepository."""

from covenant.memory.repository import MemoryRepository


async def test_episode_crud(db_session):
    repo = MemoryRepository(db_session)
    ep = await repo.add_episode(goal="test", observation="obs", action="act", outcome="ok")
    assert ep.id is not None

    fetched = await repo.get_episode(ep.id)
    assert fetched is not None
    assert fetched.goal == "test"

    episodes = await repo.list_episodes()
    assert len(episodes) == 1

    deleted = await repo.delete_episode(ep.id)
    assert deleted is True

    episodes = await repo.list_episodes()
    assert len(episodes) == 0


async def test_fact_crud(db_session):
    repo = MemoryRepository(db_session)
    fact = await repo.add_fact(subject="math", fact="2+2=4", confidence=1.0)
    assert fact.id is not None

    facts = await repo.find_facts_by_subject("math")
    assert len(facts) == 1

    updated = await repo.update_fact_confidence(fact.id, 0.8)
    assert updated is not None
    assert updated.confidence == 0.8

    deleted = await repo.delete_fact(fact.id)
    assert deleted is True


async def test_skill_crud(db_session):
    repo = MemoryRepository(db_session)
    skill = await repo.add_skill(name="calc", trigger="math question", steps=["compute"], success_criteria="correct")
    assert skill.id is not None

    by_name = await repo.get_skill_by_name("calc")
    assert by_name is not None
    assert by_name.name == "calc"

    skills = await repo.list_skills()
    assert len(skills) == 1

    deleted = await repo.delete_skill(skill.id)
    assert deleted is True


async def test_delete_nonexistent(db_session):
    repo = MemoryRepository(db_session)
    assert await repo.delete_episode(999) is False
    assert await repo.delete_fact(999) is False
    assert await repo.delete_skill(999) is False
