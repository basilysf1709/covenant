"""API route definitions."""

from __future__ import annotations

from fastapi import APIRouter

from covenant import __version__
from covenant.api.schemas import (
    HealthResponse,
    MemorySearchResponse,
    MemorySearchResult,
    RunRequest,
    RunResponse,
    StepRequest,
    StepResponse,
)
from covenant.core.agent_loop import AgentLoop
from covenant.memory.consolidation import Consolidator
from covenant.memory.database import get_session_factory
from covenant.memory.retrieval import MemoryRetriever

router = APIRouter()


def _make_agent(use_llm: bool = False) -> AgentLoop:
    kwargs = {}
    if use_llm:
        from covenant.config import get_settings
        from covenant.llm.client import get_llm_client

        settings = get_settings()
        kwargs["llm_client"] = get_llm_client(settings)
    return AgentLoop(**kwargs)


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(version=__version__)


@router.post("/step", response_model=StepResponse)
async def step(req: StepRequest):
    agent = _make_agent(use_llm=req.use_llm)
    result = await agent.step(req.user_input)
    return StepResponse(
        action=result.action.value,
        payload=result.payload,
        blocked=result.blocked,
        block_reason=result.block_reason,
    )


@router.post("/run", response_model=RunResponse)
async def run(req: RunRequest):
    agent = _make_agent(use_llm=req.use_llm)
    results = await agent.run(req.goal, max_steps=req.max_steps)
    return RunResponse(
        steps=[
            StepResponse(
                action=r.action.value,
                payload=r.payload,
                blocked=r.blocked,
                block_reason=r.block_reason,
            )
            for r in results
        ]
    )


@router.get("/memory/search", response_model=MemorySearchResponse)
async def memory_search(query: str, limit: int = 10):
    session_factory = get_session_factory()
    async with session_factory() as session:
        retriever = MemoryRetriever(session)
        results = await retriever.retrieve(query, limit=limit)
    return MemorySearchResponse(
        results=[
            MemorySearchResult(source=r.source, content=r.content, score=r.score)
            for r in results
        ]
    )


@router.post("/admin/consolidate")
async def admin_consolidate(threshold: float = 0.1):
    session_factory = get_session_factory()
    async with session_factory() as session:
        consolidator = Consolidator(session)
        report = await consolidator.run(prune_threshold=threshold)
        await session.commit()
    return {
        "deduped": report.deduped,
        "summarized": report.summarized,
        "facts_extracted": report.facts_extracted,
        "salience_updated": report.salience_updated,
        "pruned": report.pruned,
    }
