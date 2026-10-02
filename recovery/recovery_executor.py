"""
Recovery Executor – executes approved recovery actions safely.
Only actions explicitly listed in the approved allowlist can be executed.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from database.db import get_session
from database.models import RecoveryStatus
from database.repositories import JobRepository, RecoveryRepository
from recovery.checkpoint_manager import CheckpointManager
from recovery.diagnosis_agent import DiagnosisAgent
from recovery.error_classifier import ErrorClassifier
from recovery.recovery_planner import RecoveryPlan, RecoveryPlanner
from recovery.retry_manager import RetryManager
from supervisor.event_bus import EventType, bus
from supervisor.policy_engine import PolicyEngine, PolicyViolationError
from utils.logger import get_audit_logger

logger = logging.getLogger(__name__)
audit = get_audit_logger()


class RecoveryExecutor:
    """
    Orchestrates the full recovery cycle:
    Classify → Diagnose → Plan → Execute → Record.

    This is the ONLY component that executes recovery actions.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        safety_cfg = config.get("safety", {})
        retry_cfg  = config.get("retry", {})
        ckpt_cfg   = config.get("checkpoints", {})

        self._policy   = PolicyEngine(safety_cfg)
        self._retry    = RetryManager(
            max_attempts=safety_cfg.get("max_recovery_attempts", 3),
            initial_delay=retry_cfg.get("initial_delay_seconds", 5),
            backoff_multiplier=retry_cfg.get("backoff_multiplier", 2.0),
            max_delay=retry_cfg.get("max_delay_seconds", 60),
        )
        self._classifier  = ErrorClassifier()
        self._diagnoser   = DiagnosisAgent(llm_config=config.get("llm"))
        self._planner     = RecoveryPlanner(self._policy)
        self._checkpoints = CheckpointManager(
            base_dir=ckpt_cfg.get("directory", "artifacts/checkpoints"),
            max_per_job=ckpt_cfg.get("max_checkpoints_per_job", 5),
        )

    def handle_failure(self, job_id: int, error_message: str) -> bool:
        """
        Handle a job failure. Returns True if recovery succeeded, False otherwise.
        """
        if not self._retry.can_retry(job_id):
            logger.warning("Job %d: retry limit reached. No recovery.", job_id)
            bus.emit_simple(EventType.RECOVERY_FAILED, source="executor", job_id=job_id)
            return False

        attempt_number = self._retry.record_attempt(job_id)

        # ── Classify ──────────────────────────────────────────────────────────
        classified = self._classifier.classify(error_message)

        # ── Diagnose ──────────────────────────────────────────────────────────
        db = get_session()
        try:
            job_repo = JobRepository(db)
            rec_repo = RecoveryRepository(db)
            job      = job_repo.get(job_id)
            job_cfg  = job.config if job else {}

            diagnosis = self._diagnoser.diagnose(classified, job_cfg, [])

            # Record the attempt
            attempt = rec_repo.create_attempt(
                job_id=job_id,
                attempt_number=attempt_number,
                error_type=classified.error_type.value,
                error_message=error_message,
            )
            rec_repo.update_attempt(
                attempt.id,
                diagnosis=diagnosis.get("summary", ""),
            )
        finally:
            db.close()

        # ── Plan ──────────────────────────────────────────────────────────────
        if self._policy.should_stop_on_unknown_error() and classified.suggested_actions == ["stop_job"]:
            logger.error("Unknown error for job %d. Stopping safely.", job_id)
            self._record_outcome(job_id, attempt.id, "stop_job", RecoveryStatus.FAILED,
                                 "Unknown error: stopped safely.")
            bus.emit_simple(EventType.RECOVERY_FAILED, source="executor", job_id=job_id)
            return False

        plan = self._planner.plan(classified, job_cfg, attempt_number)

        # ── Execute ───────────────────────────────────────────────────────────
        try:
            self._policy.validate_action(plan.action, job_id)
            success = self._execute_action(job_id, plan, job_cfg)
        except PolicyViolationError as exc:
            logger.error("Policy blocked action for job %d: %s", job_id, exc)
            self._record_outcome(job_id, attempt.id, plan.action, RecoveryStatus.FAILED,
                                 str(exc))
            bus.emit_simple(EventType.RECOVERY_FAILED, source="executor", job_id=job_id)
            return False

        status = RecoveryStatus.SUCCESS if success else RecoveryStatus.FAILED
        self._record_outcome(job_id, attempt.id, plan.action, status, plan.rationale)

        # ── Backoff and emit result ───────────────────────────────────────────
        self._retry.wait(job_id)

        if success:
            audit.info("RECOVERY SUCCESS | job_id=%d | action=%s | attempt=%d",
                       job_id, plan.action, attempt_number)
            bus.emit_simple(EventType.RECOVERY_SUCCESS, source="executor", job_id=job_id)
            return True
        else:
            audit.warning("RECOVERY FAILED | job_id=%d | action=%s | attempt=%d",
                          job_id, plan.action, attempt_number)
            bus.emit_simple(EventType.RECOVERY_FAILED, source="executor", job_id=job_id)
            return False

    def _execute_action(self, job_id: int, plan: RecoveryPlan, job_cfg: dict) -> bool:
        action = plan.action
        logger.info("Executing recovery action '%s' for job %d.", action, job_id)

        if action == "reduce_batch_size":
            new_bs = plan.params.get("batch_size", 16)
            db = get_session()
            try:
                repo = JobRepository(db)
                job = repo.get(job_id)
                if job:
                    cfg = dict(job.config or {})
                    cfg["batch_size"] = new_bs
                    job.config = cfg
                    db.commit()
                    logger.info("Job %d batch_size reduced to %d.", job_id, new_bs)
                    return True
            finally:
                db.close()

        elif action == "clear_gpu_cache":
            try:
                import torch
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared for job %d.", job_id)
                return True
            except Exception as exc:
                logger.error("Failed to clear GPU cache: %s", exc)
                return False

        elif action == "resume_from_checkpoint":
            path = self._checkpoints.get_latest_verified(job_id)
            if path:
                logger.info("Checkpoint available for job %d: %s", job_id, path)
                return True
            logger.warning("No verified checkpoint for job %d.", job_id)
            return False

        elif action in ("restart_job", "stop_job"):
            logger.info("Action '%s' queued for job %d (orchestrator will handle).", action, job_id)
            return True

        logger.error("Unknown action '%s' – this should not happen.", action)
        return False

    def _record_outcome(
        self,
        job_id: int,
        attempt_id: int,
        action: str,
        status: RecoveryStatus,
        notes: str,
    ) -> None:
        db = get_session()
        try:
            RecoveryRepository(db).update_attempt(
                attempt_id,
                action_taken=action,
                status=status,
                outcome_notes=notes,
                resolved_at=datetime.utcnow(),
            )
        finally:
            db.close()
