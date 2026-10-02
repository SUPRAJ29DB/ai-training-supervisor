"""
Event Bus – lightweight publish/subscribe system for internal events.
Components emit events; other components subscribe to specific event types.
All callbacks are called synchronously in the emitting thread.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    # Job lifecycle
    JOB_CREATED    = "job.created"
    JOB_STARTED    = "job.started"
    JOB_PAUSED     = "job.paused"
    JOB_RESUMED    = "job.resumed"
    JOB_COMPLETED  = "job.completed"
    JOB_FAILED     = "job.failed"
    JOB_STOPPED    = "job.stopped"

    # Metrics / monitoring
    METRIC_UPDATED    = "metric.updated"
    RESOURCE_ALERT    = "resource.alert"
    JOB_STALLED       = "job.stalled"
    EPOCH_COMPLETED   = "epoch.completed"

    # Recovery
    RECOVERY_STARTED  = "recovery.started"
    RECOVERY_SUCCESS  = "recovery.success"
    RECOVERY_FAILED   = "recovery.failed"

    # Checkpoints
    CHECKPOINT_SAVED    = "checkpoint.saved"
    CHECKPOINT_RESTORED = "checkpoint.restored"

    # Evaluation
    EVALUATION_DONE = "evaluation.done"

    # Advisor
    RECOMMENDATION_READY = "recommendation.ready"

    # Notifications
    NOTIFY_USER = "notify.user"


@dataclass
class Event:
    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = ""


Handler = Callable[[Event], None]


class EventBus:
    """
    Thread-safe publish/subscribe event bus.

    Usage
    -----
    bus = EventBus()
    bus.subscribe(EventType.JOB_STARTED, my_handler)
    bus.emit(Event(type=EventType.JOB_STARTED, payload={"job_id": 1}))
    """

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[Handler]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        """Register *handler* to be called when *event_type* is emitted."""
        with self._lock:
            self._handlers[event_type].append(handler)
        logger.debug("Subscribed %s to %s", handler.__qualname__, event_type)

    def unsubscribe(self, event_type: EventType, handler: Handler) -> None:
        with self._lock:
            handlers = self._handlers.get(event_type, [])
            if handler in handlers:
                handlers.remove(handler)

    def emit(self, event: Event) -> None:
        """Publish *event* to all registered handlers (synchronous)."""
        with self._lock:
            handlers = list(self._handlers.get(event.type, []))
        for handler in handlers:
            try:
                handler(event)
            except Exception as exc:
                logger.exception(
                    "Handler %s raised an exception processing event %s: %s",
                    handler.__qualname__, event.type, exc,
                )

    def emit_simple(self, event_type: EventType, source: str = "", **payload) -> None:
        """Convenience wrapper – emit without constructing an Event manually."""
        self.emit(Event(type=event_type, payload=payload, source=source))


# Module-level singleton (imported by all components)
bus: EventBus = EventBus()
