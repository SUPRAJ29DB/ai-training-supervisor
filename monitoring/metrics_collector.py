"""
Metrics Collector – reads training metrics emitted by jobs via a shared
metrics file (JSON-lines format) and persists them in the database.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

from database.db import get_session
from database.repositories import MetricsRepository, JobRepository
from supervisor.event_bus import EventType, bus

logger = logging.getLogger(__name__)

_METRICS_FILENAME = "metrics.jsonl"


class MetricsCollector:
    """
    Watches each job's working directory for a ``metrics.jsonl`` file.
    Each line in the file should be a JSON object produced by the training script:
        {"epoch": 1, "train_loss": 0.45, "val_loss": 0.51, "accuracy": 0.82}

    The collector reads new lines, saves them to the DB, and emits METRIC_UPDATED events.
    """

    def __init__(self, interval_seconds: float = 5.0) -> None:
        self._interval = interval_seconds
        self._watched: dict[int, dict[str, Any]] = {}   # job_id → {path, offset}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="metrics-collector"
        )
        self._thread.start()
        logger.info("Metrics collector started.")

    def stop(self) -> None:
        self._running = False

    def watch_job(self, job_id: int, working_dir: str) -> None:
        """Register a job's working directory for metrics watching."""
        metrics_path = Path(working_dir) / _METRICS_FILENAME
        with self._lock:
            self._watched[job_id] = {"path": metrics_path, "offset": 0}
        logger.info("Watching metrics for job %d at %s", job_id, metrics_path)

    def unwatch_job(self, job_id: int) -> None:
        with self._lock:
            self._watched.pop(job_id, None)

    def _loop(self) -> None:
        while self._running:
            try:
                with self._lock:
                    jobs = list(self._watched.items())
                for job_id, state in jobs:
                    self._read_new_metrics(job_id, state)
            except Exception as exc:
                logger.exception("Metrics collector loop error: %s", exc)
            time.sleep(self._interval)

    def _read_new_metrics(self, job_id: int, state: dict[str, Any]) -> None:
        path: Path = state["path"]
        if not path.exists():
            return
        db = get_session()
        try:
            repo = MetricsRepository(db)
            with open(path, "r", encoding="utf-8") as fh:
                fh.seek(state["offset"])
                for raw_line in fh:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        data = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue

                    epoch = data.get("epoch", 0)
                    metric = repo.add(
                        job_id=job_id,
                        epoch=epoch,
                        step=data.get("step", 0),
                        train_loss=data.get("train_loss"),
                        val_loss=data.get("val_loss"),
                        accuracy=data.get("accuracy"),
                        extra={k: v for k, v in data.items()
                               if k not in ("epoch", "step", "train_loss", "val_loss", "accuracy")},
                    )
                    bus.emit_simple(
                        EventType.METRIC_UPDATED,
                        source="metrics_collector",
                        job_id=job_id,
                        epoch=epoch,
                        train_loss=data.get("train_loss"),
                        val_loss=data.get("val_loss"),
                        accuracy=data.get("accuracy"),
                    )
                state["offset"] = fh.tell()
        finally:
            db.close()
