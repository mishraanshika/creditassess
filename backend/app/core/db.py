"""Database session management with graceful degradation.

If Postgres is unreachable and `PERSISTENCE_REQUIRED` is false, the API keeps
serving and records decisions in a bounded in-memory ledger.  That keeps the
demo alive on a laptop with no Docker while the production path stays a single
environment variable away.
"""
from __future__ import annotations

import logging
from collections import deque
from contextlib import contextmanager
from typing import Any, Deque, Dict, Iterator, List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal: Optional[sessionmaker] = None
DB_AVAILABLE = False
DB_ERROR: str = ""


def init_engine() -> bool:
    """Create the engine and verify connectivity. Returns availability."""
    global _engine, _SessionLocal, DB_AVAILABLE, DB_ERROR
    try:
        _engine = create_engine(
            settings.DATABASE_URL,
            pool_pre_ping=True,
            pool_size=settings.DB_POOL_SIZE,
            max_overflow=settings.DB_MAX_OVERFLOW,
            future=True,
        )
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False,
                                     expire_on_commit=False, future=True)
        DB_AVAILABLE = True
        DB_ERROR = ""
        logger.info("database connected")
    except Exception as exc:  # noqa: BLE001
        DB_AVAILABLE = False
        DB_ERROR = str(exc).splitlines()[0][:300]
        if settings.PERSISTENCE_REQUIRED:
            raise
        logger.warning("database unavailable (%s); using in-memory ledger", DB_ERROR)
    return DB_AVAILABLE


@contextmanager
def session_scope() -> Iterator[Optional[Session]]:
    """Yield a session, or None when running in degraded (in-memory) mode."""
    if not DB_AVAILABLE or _SessionLocal is None:
        yield None
        return
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


class MemoryLedger:
    """Bounded, thread-safe-enough fallback store for predictions and audits."""

    def __init__(self, maxlen: int = 5000) -> None:
        self.applicants: Dict[str, Dict[str, Any]] = {}
        self.predictions: Deque[Dict[str, Any]] = deque(maxlen=maxlen)
        self.reports: Deque[Dict[str, Any]] = deque(maxlen=maxlen)
        self.audits: Deque[Dict[str, Any]] = deque(maxlen=maxlen * 2)

    def add_prediction(self, row: Dict[str, Any]) -> None:
        self.predictions.appendleft(row)

    def add_audit(self, row: Dict[str, Any]) -> None:
        self.audits.appendleft(row)

    def add_report(self, row: Dict[str, Any]) -> None:
        self.reports.appendleft(row)

    def recent_predictions(self, limit: int = 50) -> List[Dict[str, Any]]:
        return list(self.predictions)[:limit]

    def recent_audits(self, limit: int = 50) -> List[Dict[str, Any]]:
        return list(self.audits)[:limit]


ledger = MemoryLedger()
