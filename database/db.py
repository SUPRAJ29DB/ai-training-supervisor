"""
Database connection management using SQLAlchemy.
Provides session factory and init_db() for schema creation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session

from database.models import Base

logger = logging.getLogger(__name__)

_engine = None
SessionLocal: sessionmaker | None = None


def _enable_wal(dbapi_connection, connection_record):
    """Enable WAL mode for SQLite – better concurrent read performance."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def init_db(database_url: str = "sqlite:///./data/supervisor.db") -> None:
    """
    Initialise the SQLAlchemy engine and create all tables.

    Parameters
    ----------
    database_url:
        SQLAlchemy database URL.  SQLite paths are created automatically.
    """
    global _engine, SessionLocal

    if database_url.startswith("sqlite"):
        # Ensure the directory exists
        db_path = database_url.replace("sqlite:///", "").lstrip("./")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    _engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False} if "sqlite" in database_url else {},
    )

    if "sqlite" in database_url:
        event.listen(_engine, "connect", _enable_wal)

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

    # Create tables
    Base.metadata.create_all(bind=_engine)
    logger.info("Database schema created / verified at: %s", database_url)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a database session and closes it afterwards.
    """
    if SessionLocal is None:
        raise RuntimeError("Database has not been initialised. Call init_db() first.")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_session() -> Session:
    """
    Return a bare session (for use outside FastAPI request contexts).
    Remember to close it manually.
    """
    if SessionLocal is None:
        raise RuntimeError("Database has not been initialised. Call init_db() first.")
    return SessionLocal()
