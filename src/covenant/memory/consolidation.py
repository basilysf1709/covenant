"""Memory consolidation: periodic maintenance ("sleep") for the memory system."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from covenant.memory.models import Episode, Fact

log = logging.getLogger(__name__)


@dataclass
class ConsolidationReport:
    deduped: int = 0
    summarized: int = 0
    facts_extracted: int = 0
    salience_updated: int = 0
    pruned: int = 0
    errors: list[str] = field(default_factory=list)


class Consolidator:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def run(self, prune_threshold: float = 0.1) -> ConsolidationReport:
        """Execute a full consolidation pass."""
        report = ConsolidationReport()

        report.deduped = await self.dedupe_episodes()
        report.summarized = await self.summarize_clusters()
        report.facts_extracted = await self.extract_facts()
        report.salience_updated = await self.assign_salience()
        report.pruned = await self.prune_low_value(prune_threshold)

        return report

    async def dedupe_episodes(self) -> int:
        """Remove duplicate episodes based on goal+action+outcome."""
        result = await self.session.execute(
            select(Episode).order_by(Episode.ts.asc())
        )
        episodes = list(result.scalars().all())

        seen: dict[str, int] = {}
        to_delete = []
        for ep in episodes:
            key = f"{ep.goal}|{ep.action}|{ep.outcome}"
            if key in seen:
                to_delete.append(ep)
            else:
                seen[key] = ep.id

        for ep in to_delete:
            await self.session.delete(ep)

        if to_delete:
            await self.session.flush()

        return len(to_delete)

    async def summarize_clusters(self) -> int:
        """Group episodes by goal and merge clusters with >3 episodes."""
        result = await self.session.execute(
            select(Episode).order_by(Episode.ts.asc())
        )
        episodes = list(result.scalars().all())

        clusters: dict[str, list[Episode]] = defaultdict(list)
        for ep in episodes:
            clusters[ep.goal].append(ep)

        merged_count = 0
        for goal, group in clusters.items():
            if len(group) <= 3:
                continue

            # Keep the most salient episode, merge observations into it
            group.sort(key=lambda e: e.salience, reverse=True)
            keeper = group[0]
            observations = [e.observation for e in group[1:] if e.observation]
            keeper.observation = f"{keeper.observation} [+{len(observations)} merged observations]"

            for ep in group[1:]:
                await self.session.delete(ep)
                merged_count += 1

        if merged_count:
            await self.session.flush()

        return merged_count

    async def extract_facts(self) -> int:
        """Extract facts from high-confidence episodes."""
        result = await self.session.execute(
            select(Episode).where(Episode.salience >= 0.7)
        )
        episodes = list(result.scalars().all())

        extracted = 0
        for ep in episodes:
            if not ep.outcome or ep.outcome == "":
                continue

            # Check if fact already exists
            existing = await self.session.execute(
                select(Fact).where(Fact.subject == ep.goal, Fact.fact == ep.outcome)
            )
            if existing.scalar_one_or_none() is not None:
                continue

            fact = Fact(
                subject=ep.goal,
                fact=ep.outcome,
                confidence=ep.salience,
            )
            self.session.add(fact)
            extracted += 1

        if extracted:
            await self.session.flush()

        return extracted

    async def assign_salience(self) -> int:
        """Decay salience of older episodes."""
        result = await self.session.execute(
            select(Episode).where(Episode.salience > 0.1)
        )
        episodes = list(result.scalars().all())

        updated = 0
        for ep in episodes:
            new_salience = ep.salience * 0.95  # 5% decay
            if abs(new_salience - ep.salience) > 0.001:
                ep.salience = new_salience
                updated += 1

        if updated:
            await self.session.flush()

        return updated

    async def prune_low_value(self, threshold: float = 0.1) -> int:
        """Remove episodes below salience threshold."""
        result = await self.session.execute(
            select(Episode).where(Episode.salience < threshold)
        )
        episodes = list(result.scalars().all())

        for ep in episodes:
            await self.session.delete(ep)

        if episodes:
            await self.session.flush()

        return len(episodes)
