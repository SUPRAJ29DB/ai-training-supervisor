"""
Orchestrator – central coordinator of the AI Training Supervisor.
Manages the job lifecycle, delegates monitoring, recovery, and evaluation.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

from database.db import get_session
from database.models import JobStatus
from database.repositories import JobRepository, CheckpointRepository, RecoveryRepository
from supervisor.event_bus import EventBus, EventType, Event, bus as global_bus
from supervisor.job_manager import JobManager
from supervisor.job_queue import JobQueue
from supervisor.policy_engine import PolicyEngine
from supervisor.state_machine import JobStateMachine, StateMachineError
from utils.logger import get_audit_logger

logger = logging.getLogger(__name__)
audit = get_audit_logger()


class Orchestrator:
    """
    Central supervisor that:
    - Manages the job queue.
    - Starts / pauses / resumes / stops training jobs.
    - Listens to events from monitoring and recovery components.
    - Enforces safety policy on all actions.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._safety_cfg = config.get("safety", {})
        self._policy = PolicyEngine(self._safety_cfg)
        self._queue = JobQueue()
        self._log_dir = Path(config.get("app", {}).get("log_dir", "logs"))
        self._job_manager = JobManager(log_dir=self._log_dir)
        self._bus = global_bus
        self._state_machines: dict[int, JobStateMachine] = {}
        self._running = False
        self._dispatch_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Subscribe to events
        self._bus.subscribe(EventType.JOB_FAILED,    self._handle_job_failed)
        self._bus.subscribe(EventType.JOB_COMPLETED, self._handle_job_completed)
        self._bus.subscribe(EventType.RECOVERY_SUCCESS, self._handle_recovery_success)
        self._bus.subscribe(EventType.RECOVERY_FAILED,  self._handle_recovery_failed)

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background dispatcher loop."""
        self._running = True
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop, daemon=True, name="orchestrator-dispatch"
        )
        self._dispatch_thread.start()
        logger.info("Orchestrator dispatcher started.")

    def stop(self) -> None:
        """Gracefully stop the orchestrator."""
        self._running = False
        logger.info("Orchestrator stopping …")

    # ── Public API ────────────────────────────────────────────────

    def enqueue_job(self, job_id: int, name: str = "", priority: int = 0) -> None:
        """Add a job to the queue (must already exist in DB as PENDING)."""
        self._queue.push(job_id, name, priority)
        self._set_status(job_id, JobStatus.QUEUED)
        self._bus.emit_simple(EventType.JOB_CREATED, source="orchestrator", job_id=job_id)
        audit.info("JOB QUEUED | job_id=%d | name=%s | priority=%d", job_id, name, priority)

    def start_job(self, job_id: int) -> None:
        """Immediately start a specific job (bypasses queue order)."""
        self._launch_job(job_id)

    def pause_job(self, job_id: int) -> bool:
        rj = self._job_manager.get(job_id)
        if not rj:
            return False
        try:
            self._transition(job_id, JobStatus.PAUSED)
            rj.pause()
            self._bus.emit_simple(EventType.JOB_PAUSED, source="orchestrator", job_id=job_id)
            audit.info("JOB PAUSED | job_id=%d", job_id)
            return True
        except StateMachineError as exc:
            logger.warning("Cannot pause job %d: %s", job_id, exc)
            return False

    def resume_job(self, job_id: int) -> bool:
        rj = self._job_manager.get(job_id)
        if not rj:
            return False
        try:
            self._transition(job_id, JobStatus.RUNNING)
            rj.resume()
            self._bus.emit_simple(EventType.JOB_RESUMED, source="orchestrator", job_id=job_id)
            audit.info("JOB RESUMED | job_id=%d", job_id)
            return True
        except StateMachineError as exc:
            logger.warning("Cannot resume job %d: %s", job_id, exc)
            return False

    def stop_job(self, job_id: int) -> bool:
        rj = self._job_manager.get(job_id)
        if not rj and not self._queue.contains(job_id):
            return False
        self._queue.remove(job_id)
        if rj:
            rj.stop()
            self._job_manager.remove(job_id)
        self._set_status(job_id, JobStatus.STOPPED)
        self._bus.emit_simple(EventType.JOB_STOPPED, source="orchestrator", job_id=job_id)
        audit.info("JOB STOPPED | job_id=%d", job_id)
        return True

    def get_queue_snapshot(self) -> list[dict]:
        return self._queue.snapshot()

    # ── Internal helpers ──────────────────────────────────────────

    def _dispatch_loop(self) -> None:
        """Continuously pop jobs from the queue and launch them."""
        while self._running:
            try:
                active = self._job_manager.list_active()
                max_concurrent = self._config.get("training", {}).get("max_concurrent_jobs", 1)

                if len(active) < max_concurrent and not self._queue.is_empty():
                    item = self._queue.pop()
                    if item:
                        self._launch_job(item.job_id)
            except Exception as exc:
                logger.exception("Dispatch loop error: %s", exc)
            time.sleep(2)

    def _launch_job(self, job_id: int) -> None:
        """Start a single training job."""
        db = get_session()
        try:
            repo = JobRepository(db)
            job = repo.get(job_id)
            if not job:
                logger.error("Job %d not found in DB.", job_id)
                return

            job_config = job.config or {}
            script = job_config.get("script_path", "")
            working_dir = job_config.get("working_dir", ".")
            extra_args = job_config.get("args", [])

            if not script:
                logger.error("Job %d has no script_path configured.", job_id)
                self._set_status(job_id, JobStatus.FAILED, error_message="No script_path configured.")
                return

            self._transition(job_id, JobStatus.RUNNING)

            self._job_manager.launch(
                job_id=job_id,
                script_path=script,
                args=extra_args,
                working_dir=working_dir,
                on_complete=self._on_process_complete,
                on_error=self._on_process_error,
            )
            self._bus.emit_simple(EventType.JOB_STARTED, source="orchestrator", job_id=job_id)
            audit.info("JOB STARTED | job_id=%d | script=%s", job_id, script)
        finally:
            db.close()

    def _on_process_complete(self, job_id: int, returncode: int) -> None:
        self._set_status(job_id, JobStatus.COMPLETED)
        self._bus.emit_simple(EventType.JOB_COMPLETED, source="job_manager", job_id=job_id, returncode=returncode)

    def _on_process_error(self, job_id: int, error_msg: str) -> None:
        self._set_status(job_id, JobStatus.FAILED, error_message=error_msg)
        self._bus.emit_simple(EventType.JOB_FAILED, source="job_manager", job_id=job_id, error=error_msg)

    def _transition(self, job_id: int, new_status: JobStatus) -> None:
        db = get_session()
        try:
            repo = JobRepository(db)
            job = repo.get(job_id)
            if not job:
                return
            with self._lock:
                sm = self._state_machines.setdefault(job_id, JobStateMachine(job.status))
                sm.transition(new_status)
            repo.update_status(job_id, new_status)
        finally:
            db.close()

    def _set_status(self, job_id: int, status: JobStatus, **kwargs) -> None:
        db = get_session()
        try:
            JobRepository(db).update_status(job_id, status, **kwargs)
        finally:
            db.close()

    # ── Event handlers ────────────────────────────────────────────

    def _handle_job_failed(self, event: Event) -> None:
        job_id = event.payload.get("job_id")
        if not job_id:
            return
        db = get_session()
        try:
            count = RecoveryRepository(db).count_for_job(job_id)
            if not self._policy.check_retry_limit(job_id, count):
                logger.warning("Job %d exceeded recovery limit. Stopping.", job_id)
                self._set_status(job_id, JobStatus.FAILED,
                                 error_message="Recovery limit exceeded.")
        finally:
            db.close()

    def _handle_job_completed(self, event: Event) -> None:
        job_id = event.payload.get("job_id")
        logger.info("Job %d completed. Notifying user.", job_id)
        self._bus.emit_simple(
            EventType.NOTIFY_USER, source="orchestrator",
            title="Training Complete",
            message=f"Job {job_id} finished successfully.",
            level="success",
        )

    def _handle_recovery_success(self, event: Event) -> None:
        job_id = event.payload.get("job_id")
        logger.info("Recovery succeeded for job %d. Re-launching.", job_id)
        self._launch_job(job_id)

    def _handle_recovery_failed(self, event: Event) -> None:
        job_id = event.payload.get("job_id")
        logger.warning("Recovery failed for job %d. Stopping.", job_id)
        self._set_status(job_id, JobStatus.FAILED,
                         error_message="Recovery failed. Manual intervention required.")
        self._bus.emit_simple(
            EventType.NOTIFY_USER, source="orchestrator",
            title="Recovery Failed",
            message=f"Job {job_id} could not be recovered. Please review logs.",
            level="error",
        )
