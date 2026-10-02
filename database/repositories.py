"""
Data Access Layer (Repository pattern) for the AI Training Supervisor.
All database queries are centralised here to keep business logic clean.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from sqlalchemy.orm import Session

from database.models import (
    TrainingJob, TrainingMetric, Checkpoint, RecoveryAttempt,
    Experiment, EvaluationResult, Recommendation, NotificationLog,
    JobStatus, RecoveryStatus, NotificationLevel, ResourceSnapshot,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Training Job Repository
# ─────────────────────────────────────────────────────────────────

class JobRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, name: str, config: dict, **kwargs) -> TrainingJob:
        job = TrainingJob(name=name, config=config, **kwargs)
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        logger.info("Created job id=%d name=%s", job.id, job.name)
        return job

    def get(self, job_id: int) -> Optional[TrainingJob]:
        return self.db.query(TrainingJob).filter(TrainingJob.id == job_id).first()

    def list_all(self, status: Optional[JobStatus] = None) -> list[TrainingJob]:
        q = self.db.query(TrainingJob)
        if status:
            q = q.filter(TrainingJob.status == status)
        return q.order_by(TrainingJob.priority.desc(), TrainingJob.id).all()

    def update_status(self, job_id: int, status: JobStatus, **kwargs) -> Optional[TrainingJob]:
        job = self.get(job_id)
        if not job:
            return None
        job.status = status
        if status == JobStatus.RUNNING and not job.started_at:
            job.started_at = datetime.utcnow()
        if status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.STOPPED):
            job.completed_at = datetime.utcnow()
        for k, v in kwargs.items():
            setattr(job, k, v)
        self.db.commit()
        self.db.refresh(job)
        return job

    def update_epoch(self, job_id: int, epoch: int) -> None:
        job = self.get(job_id)
        if job:
            job.current_epoch = epoch
            self.db.commit()

    def delete(self, job_id: int) -> bool:
        job = self.get(job_id)
        if not job:
            return False
        self.db.delete(job)
        self.db.commit()
        return True


# ─────────────────────────────────────────────────────────────────
# Metrics Repository
# ─────────────────────────────────────────────────────────────────

class MetricsRepository:
    def __init__(self, db: Session):
        self.db = db

    def add(self, job_id: int, epoch: int, **kwargs) -> TrainingMetric:
        metric = TrainingMetric(job_id=job_id, epoch=epoch, **kwargs)
        self.db.add(metric)
        self.db.commit()
        return metric

    def get_for_job(self, job_id: int) -> list[TrainingMetric]:
        return (
            self.db.query(TrainingMetric)
            .filter(TrainingMetric.job_id == job_id)
            .order_by(TrainingMetric.epoch)
            .all()
        )

    def add_resource_snapshot(self, **kwargs) -> ResourceSnapshot:
        snap = ResourceSnapshot(**kwargs)
        self.db.add(snap)
        self.db.commit()
        return snap

    def get_recent_resources(self, limit: int = 100) -> list[ResourceSnapshot]:
        return (
            self.db.query(ResourceSnapshot)
            .order_by(ResourceSnapshot.recorded_at.desc())
            .limit(limit)
            .all()
        )


# ─────────────────────────────────────────────────────────────────
# Checkpoint Repository
# ─────────────────────────────────────────────────────────────────

class CheckpointRepository:
    def __init__(self, db: Session):
        self.db = db

    def save(self, job_id: int, epoch: int, file_path: str, **kwargs) -> Checkpoint:
        ckpt = Checkpoint(job_id=job_id, epoch=epoch, file_path=file_path, **kwargs)
        self.db.add(ckpt)
        self.db.commit()
        self.db.refresh(ckpt)
        return ckpt

    def get_latest_verified(self, job_id: int) -> Optional[Checkpoint]:
        return (
            self.db.query(Checkpoint)
            .filter(Checkpoint.job_id == job_id, Checkpoint.is_verified == True)
            .order_by(Checkpoint.epoch.desc())
            .first()
        )

    def get_all_for_job(self, job_id: int) -> list[Checkpoint]:
        return (
            self.db.query(Checkpoint)
            .filter(Checkpoint.job_id == job_id)
            .order_by(Checkpoint.epoch)
            .all()
        )

    def mark_verified(self, checkpoint_id: int) -> None:
        ckpt = self.db.query(Checkpoint).filter(Checkpoint.id == checkpoint_id).first()
        if ckpt:
            ckpt.is_verified = True
            self.db.commit()

    def delete(self, checkpoint_id: int) -> None:
        ckpt = self.db.query(Checkpoint).filter(Checkpoint.id == checkpoint_id).first()
        if ckpt:
            self.db.delete(ckpt)
            self.db.commit()


# ─────────────────────────────────────────────────────────────────
# Recovery Repository
# ─────────────────────────────────────────────────────────────────

class RecoveryRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_attempt(self, job_id: int, attempt_number: int, error_type: str,
                       error_message: str) -> RecoveryAttempt:
        attempt = RecoveryAttempt(
            job_id=job_id,
            attempt_number=attempt_number,
            error_type=error_type,
            error_message=error_message,
        )
        self.db.add(attempt)
        self.db.commit()
        self.db.refresh(attempt)
        return attempt

    def update_attempt(self, attempt_id: int, **kwargs) -> Optional[RecoveryAttempt]:
        attempt = self.db.query(RecoveryAttempt).filter(RecoveryAttempt.id == attempt_id).first()
        if not attempt:
            return None
        for k, v in kwargs.items():
            setattr(attempt, k, v)
        self.db.commit()
        return attempt

    def count_for_job(self, job_id: int) -> int:
        return self.db.query(RecoveryAttempt).filter(RecoveryAttempt.job_id == job_id).count()

    def get_for_job(self, job_id: int) -> list[RecoveryAttempt]:
        return (
            self.db.query(RecoveryAttempt)
            .filter(RecoveryAttempt.job_id == job_id)
            .order_by(RecoveryAttempt.created_at)
            .all()
        )


# ─────────────────────────────────────────────────────────────────
# Experiment Repository
# ─────────────────────────────────────────────────────────────────

class ExperimentRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, name: str, **kwargs) -> Experiment:
        exp = Experiment(name=name, **kwargs)
        self.db.add(exp)
        self.db.commit()
        self.db.refresh(exp)
        return exp

    def get(self, experiment_id: int) -> Optional[Experiment]:
        return self.db.query(Experiment).filter(Experiment.id == experiment_id).first()

    def list_all(self) -> list[Experiment]:
        return self.db.query(Experiment).order_by(Experiment.created_at.desc()).all()

    def update(self, experiment_id: int, **kwargs) -> Optional[Experiment]:
        exp = self.get(experiment_id)
        if not exp:
            return None
        for k, v in kwargs.items():
            setattr(exp, k, v)
        self.db.commit()
        return exp


# ─────────────────────────────────────────────────────────────────
# Evaluation Repository
# ─────────────────────────────────────────────────────────────────

class EvaluationRepository:
    def __init__(self, db: Session):
        self.db = db

    def save_result(self, experiment_id: int, metrics: dict, **kwargs) -> EvaluationResult:
        result = EvaluationResult(experiment_id=experiment_id, metrics=metrics, **kwargs)
        self.db.add(result)
        self.db.commit()
        self.db.refresh(result)
        return result

    def get_for_experiment(self, experiment_id: int) -> Optional[EvaluationResult]:
        return (
            self.db.query(EvaluationResult)
            .filter(EvaluationResult.experiment_id == experiment_id)
            .first()
        )


# ─────────────────────────────────────────────────────────────────
# Recommendation Repository
# ─────────────────────────────────────────────────────────────────

class RecommendationRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, title: str, **kwargs) -> Recommendation:
        rec = Recommendation(title=title, **kwargs)
        self.db.add(rec)
        self.db.commit()
        self.db.refresh(rec)
        return rec

    def list_pending(self) -> list[Recommendation]:
        return (
            self.db.query(Recommendation)
            .filter(Recommendation.status == "pending")
            .order_by(Recommendation.priority.desc(), Recommendation.created_at)
            .all()
        )

    def list_all(self) -> list[Recommendation]:
        return self.db.query(Recommendation).order_by(Recommendation.created_at.desc()).all()

    def update_status(self, rec_id: int, status: str) -> Optional[Recommendation]:
        rec = self.db.query(Recommendation).filter(Recommendation.id == rec_id).first()
        if not rec:
            return None
        rec.status = status
        rec.resolved_at = datetime.utcnow()
        self.db.commit()
        return rec


# ─────────────────────────────────────────────────────────────────
# Notification Repository
# ─────────────────────────────────────────────────────────────────

class NotificationRepository:
    def __init__(self, db: Session):
        self.db = db

    def log(self, title: str, message: str = "",
            level: NotificationLevel = NotificationLevel.INFO,
            channel: str = "desktop") -> NotificationLog:
        entry = NotificationLog(title=title, message=message, level=level, channel=channel)
        self.db.add(entry)
        self.db.commit()
        return entry

    def mark_sent(self, notification_id: int) -> None:
        entry = self.db.query(NotificationLog).filter(NotificationLog.id == notification_id).first()
        if entry:
            entry.sent = True
            self.db.commit()

    def list_recent(self, limit: int = 50) -> list[NotificationLog]:
        return (
            self.db.query(NotificationLog)
            .order_by(NotificationLog.created_at.desc())
            .limit(limit)
            .all()
        )
