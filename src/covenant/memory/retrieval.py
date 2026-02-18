"""Memory retrieval: vector + keyword + fact search with composite ranking."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from covenant.memory.models import Episode, Fact


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class RetrievalResult:
    source: str  # "episode", "fact"
    content: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryRetriever:
    def __init__(self, session: AsyncSession, embed_fn=None):
        self.session = session
        self.embed_fn = embed_fn  # async callable: str -> list[float]

    async def retrieve(self, query: str, limit: int = 10) -> list[RetrievalResult]:
        results: list[RetrievalResult] = []

        # Try vector search if embedding function is available
        if self.embed_fn is not None:
            try:
                embedding = await self.embed_fn(query)
                vector_results = await self._vector_search(embedding, limit)
                results.extend(vector_results)
            except Exception:
                pass

        # Always do keyword + fact search
        keyword_results = await self._keyword_search(query, limit)
        fact_results = await self._fact_search(query, limit)
        results.extend(keyword_results)
        results.extend(fact_results)

        ranked = self._rank(results)
        return ranked[:limit]

    async def _vector_search(self, embedding: list[float], limit: int) -> list[RetrievalResult]:
        """Cosine similarity search. Uses pgvector on Postgres, in-process on SQLite."""
        dialect = self._get_dialect()
        if dialect == "postgresql":
            return await self._vector_search_pgvector(embedding, limit)
        return await self._vector_search_python(embedding, limit)

    def _get_dialect(self) -> str:
        bind = self.session.get_bind()
        return bind.dialect.name

    async def _vector_search_pgvector(self, embedding: list[float], limit: int) -> list[RetrievalResult]:
        try:
            stmt = text(
                "SELECT id, goal, observation, action, outcome, salience, "
                "1 - (embedding <=> :emb::vector) as similarity "
                "FROM episodes WHERE embedding IS NOT NULL "
                "ORDER BY embedding <=> :emb::vector LIMIT :lim"
            )
            result = await self.session.execute(stmt, {"emb": str(embedding), "lim": limit})
            rows = result.fetchall()
            return [
                RetrievalResult(
                    source="episode",
                    content=f"{row.goal}: {row.observation} -> {row.action} -> {row.outcome}",
                    score=float(row.similarity) if row.similarity else 0.0,
                    metadata={"id": row.id, "salience": float(row.salience)},
                )
                for row in rows
            ]
        except Exception:
            return []

    async def _vector_search_python(self, embedding: list[float], limit: int) -> list[RetrievalResult]:
        """In-process cosine similarity for SQLite (no pgvector)."""
        try:
            stmt = select(Episode).where(Episode.embedding.is_not(None))
            result = await self.session.execute(stmt)
            episodes = result.scalars().all()

            scored: list[tuple[float, Episode]] = []
            for ep in episodes:
                try:
                    stored = json.loads(ep.embedding)
                except (json.JSONDecodeError, TypeError):
                    continue
                sim = _cosine_similarity(embedding, stored)
                scored.append((sim, ep))

            scored.sort(key=lambda t: t[0], reverse=True)
            return [
                RetrievalResult(
                    source="episode",
                    content=f"{ep.goal}: {ep.observation} -> {ep.action} -> {ep.outcome}",
                    score=sim,
                    metadata={"id": ep.id, "salience": ep.salience},
                )
                for sim, ep in scored[:limit]
            ]
        except Exception:
            return []

    async def _keyword_search(self, query: str, limit: int) -> list[RetrievalResult]:
        """ILIKE-based keyword search fallback."""
        pattern = f"%{query}%"
        stmt = (
            select(Episode)
            .where(
                Episode.goal.ilike(pattern)
                | Episode.observation.ilike(pattern)
                | Episode.action.ilike(pattern)
                | Episode.outcome.ilike(pattern)
            )
            .order_by(Episode.ts.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        episodes = result.scalars().all()
        return [
            RetrievalResult(
                source="episode",
                content=f"{ep.goal}: {ep.observation} -> {ep.action} -> {ep.outcome}",
                score=0.3,  # base score for keyword match
                metadata={"id": ep.id, "salience": ep.salience},
            )
            for ep in episodes
        ]

    async def _fact_search(self, query: str, limit: int) -> list[RetrievalResult]:
        """Subject-based fact lookup."""
        pattern = f"%{query}%"
        stmt = (
            select(Fact)
            .where(Fact.subject.ilike(pattern) | Fact.fact.ilike(pattern))
            .order_by(Fact.confidence.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        facts = result.scalars().all()
        return [
            RetrievalResult(
                source="fact",
                content=f"{f.subject}: {f.fact}",
                score=float(f.confidence) * 0.5,
                metadata={"id": f.id, "confidence": f.confidence},
            )
            for f in facts
        ]

    def _rank(
        self,
        results: list[RetrievalResult],
        recency_w: float = 0.3,
        salience_w: float = 0.3,
        similarity_w: float = 0.4,
    ) -> list[RetrievalResult]:
        """Composite scoring: combine similarity, salience, and recency."""
        for r in results:
            salience = r.metadata.get("salience", 0.5)
            composite = (similarity_w * r.score) + (salience_w * salience) + (recency_w * 0.5)
            r.score = composite
        results.sort(key=lambda r: r.score, reverse=True)
        return results
