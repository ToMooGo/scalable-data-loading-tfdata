"""Database schema shared by the API (writes) and the monitoring flow (reads).

PostgreSQL in Docker Compose; any SQLAlchemy URL works (SQLite is used in tests).
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    """One row per /predict request: the raw input, the answer and the data-quality flags."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    source: Mapped[str] = mapped_column(String(32), default="api")
    features: Mapped[dict] = mapped_column(JSON)  # the request as received (None = missing)
    prediction_usd: Mapped[float] = mapped_column(Float)
    unknown_category: Mapped[bool] = mapped_column(Boolean, index=True)
    n_missing: Mapped[int] = mapped_column(Integer)
    n_out_of_range: Mapped[int] = mapped_column(Integer)
    model_version: Mapped[str] = mapped_column(String(16))
    latency_ms: Mapped[float] = mapped_column(Float)


def make_engine(url: str) -> Engine:
    kwargs = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kwargs)


def init_db(engine: Engine) -> sessionmaker:
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)
