"""
Health Checker – detects stalled training jobs.
A job is considered stalled if no new metric has been logged within the
configured stall_timeout_seconds window.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any

from database.db import get_session
from database.repositories import MetricsRepository, JobRepository
from database.models import JobStatus
from supervisor.event_bus import EventType, bus

logger = logging.getLogger(__name__)


class HealthChecker:
    """
    Periodically queries the DB for last metric timestamps of running jobs.
    Emits JOB_STALLED if a job has had no metric update past the timeout.
    """

    def __init__(self, config: dict[str, Any], check_interval: float = 30.0) -> None:
        self._timeout = timedelta(
            seconds=config.get("resources", {}).get("stall_timeout_seconds", 300)
        )
        self._interval = check_interval
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="health-checker"
        )
        self._thread.start()
        logger.info("Health checker started (stall timeout=%s).", self._timeout)

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        while self._running:
            try:
                self._check()
            except Exception as exc:
                logger.exception("Health checker error: %s", exc)
            time.sleep(self._interval)

    def _check(self) -> None:
        db = get_session()
        try:
            job_repo = JobRepository(db)
            metrics_repo = MetricsRepository(db)

            running_jobs = job_repo.list_all(status=JobStatus.RUNNING)
            now = datetime.utcnow()

            for job in running_jobs:
                metrics = metrics_repo.get_for_job(job.id)
                if not metrics:
                    # No metrics yet – check how long the job has been running
                    if job.started_at and (now - job.started_at) > self._timeout:
                        self._emit_stall(job.id, "No metrics received since job started.")
                    continue

                last_metric = metrics[-1]
                if (now - last_metric.recorded_at) > self._timeout:
                    self._emit_stall(
                        job.id,
                        f"No metric update in {self._timeout}. Last epoch={last_metric.epoch}.",
                    )
        finally:
            db.close()

    def _emit_stall(self, job_id: int, reason: str) -> None:
        logger.warning("Job %d appears stalled: %s", job_id, reason)
        bus.emit_simple(
            EventType.JOB_STALLED,
            source="health_checker",
            job_id=job_id,
            reason=reason,
        )
