"""CRUD repository for memory models."""

import json
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from covenant.memory.models import Episode, Fact, Skill


class MemoryRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    # --- Episode ---
    async def add_episode(self, **kwargs) -> Episode:
        ep = Episode(**kwargs)
        self.session.add(ep)
        await self.session.flush()
        return ep

    async def get_episode(self, episode_id: int) -> Episode | None:
        return await self.session.get(Episode, episode_id)

    async def list_episodes(self, limit: int = 50, offset: int = 0) -> list[Episode]:
        result = await self.session.execute(
            select(Episode).order_by(Episode.ts.desc()).limit(limit).offset(offset)
        )
        return list(result.scalars().all())

    async def add_episode_with_embedding(self, embedding: list[float] | None = None, **kwargs) -> Episode:
        ep = Episode(**kwargs)
        if embedding is not None:
            ep.embedding = json.dumps(embedding)
        self.session.add(ep)
        await self.session.flush()
        return ep

    async def delete_episode(self, episode_id: int) -> bool:
        ep = await self.get_episode(episode_id)
        if ep is None:
            return False
        await self.session.delete(ep)
        await self.session.flush()
        return True

    # --- Fact ---
    async def add_fact(self, **kwargs) -> Fact:
        fact = Fact(**kwargs)
        self.session.add(fact)
        await self.session.flush()
        return fact

    async def get_fact(self, fact_id: int) -> Fact | None:
        return await self.session.get(Fact, fact_id)

    async def find_facts_by_subject(self, subject: str) -> list[Fact]:
        result = await self.session.execute(
            select(Fact).where(Fact.subject == subject)
        )
        return list(result.scalars().all())

    async def update_fact_confidence(self, fact_id: int, confidence: float) -> Fact | None:
        fact = await self.get_fact(fact_id)
        if fact is None:
            return None
        fact.confidence = confidence
        fact.last_verified_ts = datetime.now(timezone.utc)
        await self.session.flush()
        return fact

    async def delete_fact(self, fact_id: int) -> bool:
        fact = await self.get_fact(fact_id)
        if fact is None:
            return False
        await self.session.delete(fact)
        await self.session.flush()
        return True

    # --- Skill ---
    async def add_skill(self, **kwargs) -> Skill:
        skill = Skill(**kwargs)
        self.session.add(skill)
        await self.session.flush()
        return skill

    async def get_skill(self, skill_id: int) -> Skill | None:
        return await self.session.get(Skill, skill_id)

    async def get_skill_by_name(self, name: str) -> Skill | None:
        result = await self.session.execute(
            select(Skill).where(Skill.name == name)
        )
        return result.scalar_one_or_none()

    async def list_skills(self) -> list[Skill]:
        result = await self.session.execute(select(Skill).order_by(Skill.name))
        return list(result.scalars().all())

    async def delete_skill(self, skill_id: int) -> bool:
        skill = await self.get_skill(skill_id)
        if skill is None:
            return False
        await self.session.delete(skill)
        await self.session.flush()
        return True
