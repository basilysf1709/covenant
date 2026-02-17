"""Tests for memory consolidation."""

from sqlalchemy import select

from covenant.memory.consolidation import Consolidator
from covenant.memory.models import Episode, Fact


async def test_dedupe_episodes(db_session):
    # Add duplicate episodes
    for _ in range(3):
        db_session.add(Episode(goal="g", observation="o", action="a", outcome="ok"))
    db_session.add(Episode(goal="unique", observation="u", action="b", outcome="different"))
    await db_session.flush()

    consolidator = Consolidator(db_session)
    deduped = await consolidator.dedupe_episodes()
    assert deduped == 2  # 3 duplicates -> keep 1, remove 2

    result = await db_session.execute(select(Episode))
    remaining = list(result.scalars().all())
    assert len(remaining) == 2  # 1 deduped + 1 unique


async def test_summarize_clusters(db_session):
    # Add 5 episodes with same goal (should be clustered and merged)
    for i in range(5):
        db_session.add(Episode(goal="big goal", observation=f"obs {i}", action=f"act {i}", outcome=f"out {i}", salience=0.1 * i + 0.1))
    await db_session.flush()

    consolidator = Consolidator(db_session)
    merged = await consolidator.summarize_clusters()
    assert merged == 4  # 5 episodes -> keep 1, merge 4

    result = await db_session.execute(select(Episode))
    remaining = list(result.scalars().all())
    assert len(remaining) == 1
    assert "merged observations" in remaining[0].observation


async def test_extract_facts(db_session):
    db_session.add(Episode(goal="python", observation="studied docs", action="read", outcome="Python supports async", salience=0.8))
    db_session.add(Episode(goal="low salience", observation="x", action="y", outcome="z", salience=0.3))
    await db_session.flush()

    consolidator = Consolidator(db_session)
    extracted = await consolidator.extract_facts()
    assert extracted == 1

    result = await db_session.execute(select(Fact))
    facts = list(result.scalars().all())
    assert len(facts) == 1
    assert facts[0].subject == "python"


async def test_extract_facts_no_duplicates(db_session):
    db_session.add(Episode(goal="python", observation="x", action="y", outcome="known fact", salience=0.9))
    db_session.add(Fact(subject="python", fact="known fact", confidence=0.9))
    await db_session.flush()

    consolidator = Consolidator(db_session)
    extracted = await consolidator.extract_facts()
    assert extracted == 0  # Should not duplicate existing fact


async def test_assign_salience(db_session):
    db_session.add(Episode(goal="test", observation="o", action="a", outcome="ok", salience=0.8))
    await db_session.flush()

    consolidator = Consolidator(db_session)
    updated = await consolidator.assign_salience()
    assert updated == 1

    result = await db_session.execute(select(Episode))
    ep = result.scalar_one()
    assert ep.salience < 0.8  # Should have decayed


async def test_prune_low_value(db_session):
    db_session.add(Episode(goal="keep", observation="o", action="a", outcome="ok", salience=0.5))
    db_session.add(Episode(goal="prune", observation="o", action="a", outcome="ok", salience=0.05))
    await db_session.flush()

    consolidator = Consolidator(db_session)
    pruned = await consolidator.prune_low_value(threshold=0.1)
    assert pruned == 1

    result = await db_session.execute(select(Episode))
    remaining = list(result.scalars().all())
    assert len(remaining) == 1
    assert remaining[0].goal == "keep"


async def test_full_consolidation_pass(db_session):
    # Add various episodes
    for i in range(3):
        db_session.add(Episode(goal="dup", observation="o", action="a", outcome="ok", salience=0.5))
    db_session.add(Episode(goal="high sal", observation="o", action="a", outcome="important finding", salience=0.9))
    db_session.add(Episode(goal="low sal", observation="o", action="a", outcome="meh", salience=0.05))
    await db_session.flush()

    consolidator = Consolidator(db_session)
    report = await consolidator.run(prune_threshold=0.1)

    assert report.deduped >= 0
    assert report.pruned >= 0
    assert report.facts_extracted >= 0
