"""Pydantic schemas for the API layer."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class StepRequest(BaseModel):
    user_input: str
    use_llm: bool = False


class StepResponse(BaseModel):
    action: str
    payload: dict[str, Any] = {}
    blocked: bool = False
    block_reason: str = ""


class RunRequest(BaseModel):
    goal: str
    max_steps: int = 10
    use_llm: bool = False


class RunResponse(BaseModel):
    steps: list[StepResponse]


class MemorySearchRequest(BaseModel):
    query: str
    limit: int = 10


class MemorySearchResult(BaseModel):
    source: str
    content: str
    score: float


class MemorySearchResponse(BaseModel):
    results: list[MemorySearchResult]


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
