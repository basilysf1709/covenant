"""Tests for memory retrieval system."""

import json

from covenant.memory.models import Episode, Fact
from covenant.memory.retrieval import MemoryRetriever, RetrievalResult, _cosine_similarity


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
    """Test that embed_fn is called and vector search works on SQLite."""
    call_count = 0

    async def fake_embed(text: str) -> list[float]:
        nonlocal call_count
        call_count += 1
        return [0.1] * 4

    # Episode WITH embedding — should be found by vector search
    db_session.add(Episode(
        goal="test", observation="obs", action="act", outcome="ok",
        embedding=json.dumps([0.1] * 4),
    ))
    # Episode WITHOUT embedding — only reachable via keyword
    db_session.add(Episode(
        goal="other", observation="other", action="other", outcome="other",
    ))
    await db_session.flush()

    retriever = MemoryRetriever(db_session, embed_fn=fake_embed)
    results = await retriever.retrieve("test", limit=5)
    assert call_count == 1  # embed_fn was called
    # Should get both vector and keyword results
    assert len(results) >= 1


async def test_vector_search_sqlite(db_session):
    """SQLite vector search returns results ranked by cosine similarity."""
    emb_a = [1.0, 0.0, 0.0]
    emb_b = [0.0, 1.0, 0.0]
    query_emb = [0.9, 0.1, 0.0]  # closer to emb_a

    db_session.add(Episode(goal="alpha", observation="a", action="a", outcome="a", embedding=json.dumps(emb_a)))
    db_session.add(Episode(goal="beta", observation="b", action="b", outcome="b", embedding=json.dumps(emb_b)))
    await db_session.flush()

    async def embed_fn(text: str) -> list[float]:
        return query_emb

    retriever = MemoryRetriever(db_session, embed_fn=embed_fn)
    results = await retriever.retrieve("anything", limit=5)
    # Vector results should rank alpha higher than beta
    vector_results = [r for r in results if r.source == "episode"]
    assert len(vector_results) >= 2
    # alpha should appear before beta (higher similarity)
    alpha_idx = next(i for i, r in enumerate(vector_results) if "alpha" in r.content)
    beta_idx = next(i for i, r in enumerate(vector_results) if "beta" in r.content)
    assert alpha_idx < beta_idx


def test_cosine_similarity():
    """Unit test for the cosine similarity helper."""
    assert _cosine_similarity([1, 0, 0], [1, 0, 0]) == 1.0
    assert _cosine_similarity([1, 0, 0], [0, 1, 0]) == 0.0
    assert _cosine_similarity([0, 0, 0], [1, 0, 0]) == 0.0
    assert abs(_cosine_similarity([1, 1], [1, 0]) - 0.7071) < 0.01
