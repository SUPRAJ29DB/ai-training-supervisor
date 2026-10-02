"""
Retry Manager – tracks retry attempts per job and implements exponential backoff.
"""

from __future__ import annotations

import logging
import time
import threading
from typing import Any

logger = logging.getLogger(__name__)


class RetryManager:
    """
    Tracks the number of recovery attempts per job and computes backoff delays.

    Parameters
    ----------
    max_attempts:        Global maximum retries per job.
    initial_delay:       Seconds to wait before first retry.
    backoff_multiplier:  Multiplier applied after each failure.
    max_delay:           Cap on delay (seconds).
    """

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 5.0,
        backoff_multiplier: float = 2.0,
        max_delay: float = 60.0,
    ) -> None:
        self._max_attempts = max_attempts
        self._initial_delay = initial_delay
        self._backoff = backoff_multiplier
        self._max_delay = max_delay
        self._attempts: dict[int, int] = {}        # job_id → count
        self._lock = threading.Lock()

    def can_retry(self, job_id: int) -> bool:
        with self._lock:
            return self._attempts.get(job_id, 0) < self._max_attempts

    def record_attempt(self, job_id: int) -> int:
        """Increment attempt count and return the new count."""
        with self._lock:
            self._attempts[job_id] = self._attempts.get(job_id, 0) + 1
            return self._attempts[job_id]

    def attempt_count(self, job_id: int) -> int:
        with self._lock:
            return self._attempts.get(job_id, 0)

    def reset(self, job_id: int) -> None:
        with self._lock:
            self._attempts.pop(job_id, None)

    def backoff_delay(self, job_id: int) -> float:
        count = self.attempt_count(job_id)
        delay = min(self._initial_delay * (self._backoff ** (count - 1)), self._max_delay)
        return delay

    def wait(self, job_id: int) -> None:
        delay = self.backoff_delay(job_id)
        logger.info("Backoff: waiting %.1f s before retry for job %d.", delay, job_id)
        time.sleep(delay)
