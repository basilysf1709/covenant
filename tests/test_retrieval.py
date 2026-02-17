"""Tests for memory retrieval system."""

from covenant.memory.models import Episode, Fact
from covenant.memory.retrieval import MemoryRetriever, RetrievalResult


async def test_keyword_search(db_session):
    # Add test episodes
    db_session.add(Episode(goal="learn python", observation="read docs", action="study", outcome="understood basics"))
    db_session.add(Episode(goal="cook dinner", observation="checked fridge", action="cooked pasta", outcome="tasty"))
    await db_session.flush()

    retriever = MemoryRetriever(db_session)
    results = await retriever.retrieve("python", limit=5)
    assert len(results) >= 1
    assert any("python" in r.content.lower() for r in results)


async def test_fact_search(db_session):
    db_session.add(Fact(subject="python", fact="Python is a programming language", confidence=0.9))
    db_session.add(Fact(subject="java", fact="Java is also a programming language", confidence=0.8))
    await db_session.flush()

    retriever = MemoryRetriever(db_session)
    results = await retriever.retrieve("python", limit=5)
    assert len(results) >= 1
    assert any("python" in r.content.lower() for r in results)


async def test_ranking():
    retriever = MemoryRetriever(session=None)  # session not needed for ranking
    results = [
        RetrievalResult(source="episode", content="a", score=0.9, metadata={"salience": 0.8}),
        RetrievalResult(source="episode", content="b", score=0.2, metadata={"salience": 0.3}),
        RetrievalResult(source="fact", content="c", score=0.5, metadata={"salience": 0.5, "confidence": 0.7}),
    ]
    ranked = retriever._rank(results)
    # Highest composite score should be first
    assert ranked[0].content == "a"


async def test_empty_retrieval(db_session):
    retriever = MemoryRetriever(db_session)
    results = await retriever.retrieve("nonexistent query xyz")
    assert results == []


async def test_retrieve_with_embed_fn(db_session):
    """Test that embed_fn is called but vector search gracefully fails on SQLite."""
    call_count = 0

    async def fake_embed(text: str) -> list[float]:
        nonlocal call_count
        call_count += 1
        return [0.0] * 1536

    db_session.add(Episode(goal="test", observation="obs", action="act", outcome="ok"))
    await db_session.flush()

    retriever = MemoryRetriever(db_session, embed_fn=fake_embed)
    results = await retriever.retrieve("test", limit=5)
    assert call_count == 1  # embed_fn was called
    # Should still get keyword results even if vector search fails
    assert len(results) >= 1
