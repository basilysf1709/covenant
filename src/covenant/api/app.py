"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from covenant import __version__
from covenant.config import get_settings
from covenant.memory.database import close_db, init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    await init_db(settings)
    yield
    await close_db()


def create_app() -> FastAPI:
    app = FastAPI(title="Covenant", version=__version__, lifespan=lifespan)

    from covenant.api.routes import router
    app.include_router(router)

    return app
