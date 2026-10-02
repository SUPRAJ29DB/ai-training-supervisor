"""
Progress Tracker – tracks epoch/batch progress for running jobs
and updates the database.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from database.db import get_session
from database.repositories import JobRepository
from supervisor.event_bus import EventType, bus

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Listens to EPOCH_COMPLETED events and updates job progress in the DB."""

    def __init__(self) -> None:
        self._progress: dict[int, dict[str, Any]] = {}
        self._lock = threading.Lock()
        bus.subscribe(EventType.EPOCH_COMPLETED, self._on_epoch_completed)

    def _on_epoch_completed(self, event) -> None:
        job_id = event.payload.get("job_id")
        current = event.payload.get("current_epoch", 0)
        total   = event.payload.get("total_epochs", 0)

        with self._lock:
            self._progress[job_id] = {
                "current_epoch": current,
                "total_epochs":  total,
                "percent":       round(current / total * 100, 1) if total else 0,
            }

        # Persist to DB
        db = get_session()
        try:
            repo = JobRepository(db)
            job = repo.get(job_id)
            if job:
                job.current_epoch = current
                job.total_epochs  = total
                db.commit()
        except Exception as exc:
            logger.exception("Progress tracker DB update failed: %s", exc)
        finally:
            db.close()

    def get_progress(self, job_id: int) -> dict[str, Any]:
        with self._lock:
            return self._progress.get(job_id, {"current_epoch": 0, "total_epochs": 0, "percent": 0})
