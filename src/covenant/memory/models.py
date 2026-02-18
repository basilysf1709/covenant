"""SQLAlchemy ORM models for Covenant's memory system."""

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Episode(Base):
    __tablename__ = "episodes"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    goal: Mapped[str] = mapped_column(Text, default="")
    observation: Mapped[str] = mapped_column(Text, default="")
    action: Mapped[str] = mapped_column(Text, default="")
    outcome: Mapped[str] = mapped_column(Text, default="")
    salience: Mapped[float] = mapped_column(Float, default=0.5)
    tags: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=None)
    embedding: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)

    def __repr__(self) -> str:
        return f"<Episode(id={self.id}, goal={self.goal!r})>"


class Fact(Base):
    __tablename__ = "facts"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ts_created: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    subject: Mapped[str] = mapped_column(String(255), index=True)
    fact: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    last_verified_ts: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )

    def __repr__(self) -> str:
        return f"<Fact(id={self.id}, subject={self.subject!r})>"


class Skill(Base):
    __tablename__ = "skills"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    trigger: Mapped[str] = mapped_column(Text, default="")
    steps: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=None)
    constraints: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=None)
    success_criteria: Mapped[str] = mapped_column(Text, default="")
    # embedding column added later via Alembic migration

    def __repr__(self) -> str:
        return f"<Skill(id={self.id}, name={self.name!r})>"
