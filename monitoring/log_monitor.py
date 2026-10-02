"""
Log Monitor – tails job log files and extracts structured events
(errors, warnings, epoch completions).
"""

from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path
from typing import Optional

from supervisor.event_bus import EventType, bus

logger = logging.getLogger(__name__)

# Patterns that indicate training errors
ERROR_PATTERNS = [
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"RuntimeError", re.IGNORECASE),
    re.compile(r"FileNotFoundError", re.IGNORECASE),
    re.compile(r"KeyError", re.IGNORECASE),
    re.compile(r"ValueError", re.IGNORECASE),
    re.compile(r"Traceback \(most recent call last\)", re.IGNORECASE),
    re.compile(r"NaN loss detected", re.IGNORECASE),
]

EPOCH_PATTERN = re.compile(
    r"[Ee]poch\s+(\d+)[\s/]+(\d+)", re.IGNORECASE
)


class LogMonitor:
    """
    Background monitor that tails job log files, detects errors,
    and emits appropriate bus events.
    """

    def __init__(self, interval_seconds: float = 3.0) -> None:
        self._interval = interval_seconds
        self._watched: dict[int, dict] = {}    # job_id → {path, offset}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="log-monitor"
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def watch_job(self, job_id: int, log_path: Path) -> None:
        with self._lock:
            self._watched[job_id] = {"path": log_path, "offset": 0, "errors": []}

    def unwatch_job(self, job_id: int) -> None:
        with self._lock:
            self._watched.pop(job_id, None)

    def _loop(self) -> None:
        while self._running:
            try:
                with self._lock:
                    items = list(self._watched.items())
                for job_id, state in items:
                    self._process_new_lines(job_id, state)
            except Exception as exc:
                logger.exception("Log monitor loop error: %s", exc)
            time.sleep(self._interval)

    def _process_new_lines(self, job_id: int, state: dict) -> None:
        log_path: Path = state["path"]
        if not log_path.exists():
            return
        with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
            fh.seek(state["offset"])
            for line in fh:
                self._analyse_line(job_id, line, state)
            state["offset"] = fh.tell()

    def _analyse_line(self, job_id: int, line: str, state: dict) -> None:
        # Check for errors
        for pattern in ERROR_PATTERNS:
            if pattern.search(line):
                state["errors"].append(line.strip())
                bus.emit_simple(
                    EventType.JOB_FAILED,
                    source="log_monitor",
                    job_id=job_id,
                    error=line.strip(),
                )
                logger.warning("Error detected in job %d log: %s", job_id, line.strip())
                return

        # Check for epoch completions
        m = EPOCH_PATTERN.search(line)
        if m:
            current_epoch = int(m.group(1))
            total_epochs  = int(m.group(2))
            bus.emit_simple(
                EventType.EPOCH_COMPLETED,
                source="log_monitor",
                job_id=job_id,
                current_epoch=current_epoch,
                total_epochs=total_epochs,
            )
