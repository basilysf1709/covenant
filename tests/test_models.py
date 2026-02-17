"""Tests for memory ORM models."""

from covenant.memory.models import Episode, Fact, Skill


async def test_create_episode(db_session):
    ep = Episode(goal="test goal", observation="saw something", action="did something", outcome="success")
    db_session.add(ep)
    await db_session.flush()
    assert ep.id is not None
    assert ep.salience == 0.5
    assert ep.ts is not None


async def test_create_fact(db_session):
    fact = Fact(subject="python", fact="Python is a programming language", confidence=0.9)
    db_session.add(fact)
    await db_session.flush()
    assert fact.id is not None
    assert fact.confidence == 0.9


async def test_create_skill(db_session):
    skill = Skill(
        name="greeting",
        trigger="user says hello",
        steps=["wave", "say hi"],
        success_criteria="user responds positively",
    )
    db_session.add(skill)
    await db_session.flush()
    assert skill.id is not None
    assert skill.name == "greeting"
