"""Tests for the FastAPI API layer."""

import pytest
from httpx import ASGITransport, AsyncClient

import covenant.memory.database as db_mod
from covenant.api.app import create_app
from covenant.config import Settings
from covenant.memory.database import close_db, init_db


@pytest.fixture(autouse=True)
async def _setup_db():
    """Initialize in-memory DB before each test, clean up after."""
    db_mod._engine = None
    db_mod._session_factory = None
    settings = Settings(database_url="sqlite+aiosqlite://")
    await init_db(settings)
    yield
    await close_db()


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


async def test_step(client):
    resp = await client.post("/step", json={"user_input": "hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert "action" in data
    assert "payload" in data


async def test_run(client):
    resp = await client.post("/run", json={"goal": "test goal", "max_steps": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert "steps" in data
    assert len(data["steps"]) >= 1


async def test_memory_search(client):
    resp = await client.get("/memory/search", params={"query": "test", "limit": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
