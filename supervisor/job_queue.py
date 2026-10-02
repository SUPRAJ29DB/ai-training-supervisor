"""
Job Queue – priority queue for managing pending/queued training jobs.
Thread-safe, supports priority ordering.
"""

from __future__ import annotations

import heapq
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class QueueItem:
    priority: int
    job_id: int
    name: str = ""

    def __lt__(self, other: "QueueItem") -> bool:
        # Higher priority first.
        # For equal priority, lower job_id first (FIFO).
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.job_id < other.job_id


class JobQueue:
    """
    Thread-safe priority queue for training job IDs.

    Items are popped in descending priority order (highest first).
    If priorities are equal, FIFO ordering is preserved via job_id.
    """

    def __init__(self) -> None:
        self._heap: list[QueueItem] = []
        self._lock = threading.Lock()
        self._in_queue: set[int] = set()

    def push(self, job_id: int, name: str = "", priority: int = 0) -> None:
        with self._lock:
            if job_id in self._in_queue:
                logger.warning("Job %d already in queue – skipping duplicate.", job_id)
                return
            item = QueueItem(priority=priority, job_id=job_id, name=name)
            heapq.heappush(self._heap, item)
            self._in_queue.add(job_id)
        logger.debug("Queued job id=%d priority=%d", job_id, priority)

    def pop(self) -> Optional[QueueItem]:
        with self._lock:
            if not self._heap:
                return None
            item = heapq.heappop(self._heap)
            self._in_queue.discard(item.job_id)
        logger.debug("Dequeued job id=%d", item.job_id)
        return item

    def remove(self, job_id: int) -> bool:
        """Remove a job from the queue without popping it (e.g., when stopped)."""
        with self._lock:
            if job_id not in self._in_queue:
                return False
            self._heap = [i for i in self._heap if i.job_id != job_id]
            heapq.heapify(self._heap)
            self._in_queue.discard(job_id)
        logger.info("Removed job id=%d from queue.", job_id)
        return True

    def peek(self) -> Optional[QueueItem]:
        with self._lock:
            return self._heap[0] if self._heap else None

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._heap) == 0

    def size(self) -> int:
        with self._lock:
            return len(self._heap)

    def contains(self, job_id: int) -> bool:
        with self._lock:
            return job_id in self._in_queue

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [
                {"job_id": item.job_id, "name": item.name, "priority": item.priority}
                for item in sorted(self._heap)
            ]
