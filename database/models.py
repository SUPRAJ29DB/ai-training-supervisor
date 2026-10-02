"""
SQLAlchemy ORM models for the AI Training Supervisor.
All tables are defined here; the database schema is auto-created by init_db().
"""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text,
    ForeignKey, Enum as SAEnum, JSON,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────

class JobStatus(str, enum.Enum):
    PENDING   = "pending"
    QUEUED    = "queued"
    RUNNING   = "running"
    PAUSED    = "paused"
    COMPLETED = "completed"
    FAILED    = "failed"
    STOPPED   = "stopped"
    RECOVERING = "recovering"


class RecoveryStatus(str, enum.Enum):
    PENDING  = "pending"
    SUCCESS  = "success"
    FAILED   = "failed"
    SKIPPED  = "skipped"


class NotificationLevel(str, enum.Enum):
    INFO    = "info"
    WARNING = "warning"
    ERROR   = "error"
    SUCCESS = "success"


# ─────────────────────────────────────────────────────────────────
# Training Job
# ─────────────────────────────────────────────────────────────────

class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id            = Column(Integer, primary_key=True, index=True)
    name          = Column(String(255), nullable=False)
    description   = Column(Text, default="")
    framework     = Column(String(50), default="pytorch")   # pytorch | tensorflow | sklearn | custom
    model_type    = Column(String(100), default="classifier")
    status        = Column(SAEnum(JobStatus), default=JobStatus.PENDING, nullable=False)
    config        = Column(JSON, default=dict)               # Hyperparams, paths, etc.
    working_dir   = Column(String(512), default="")
    script_path   = Column(String(512), default="")
    created_at    = Column(DateTime, default=datetime.utcnow)
    started_at    = Column(DateTime, nullable=True)
    completed_at  = Column(DateTime, nullable=True)
    error_message = Column(Text, default="")
    current_epoch = Column(Integer, default=0)
    total_epochs  = Column(Integer, default=0)
    priority      = Column(Integer, default=0)              # Higher = earlier in queue

    metrics     = relationship("TrainingMetric",   back_populates="job", cascade="all, delete-orphan")
    checkpoints = relationship("Checkpoint",       back_populates="job", cascade="all, delete-orphan")
    recoveries  = relationship("RecoveryAttempt",  back_populates="job", cascade="all, delete-orphan")
    experiment  = relationship("Experiment",       back_populates="job", uselist=False)


# ─────────────────────────────────────────────────────────────────
# Training Metrics
# ─────────────────────────────────────────────────────────────────

class TrainingMetric(Base):
    __tablename__ = "training_metrics"

    id          = Column(Integer, primary_key=True, index=True)
    job_id      = Column(Integer, ForeignKey("training_jobs.id"), nullable=False)
    epoch       = Column(Integer, nullable=False)
    step        = Column(Integer, default=0)
    train_loss  = Column(Float, nullable=True)
    val_loss    = Column(Float, nullable=True)
    accuracy    = Column(Float, nullable=True)
    extra       = Column(JSON, default=dict)    # Task-specific metrics
    recorded_at = Column(DateTime, default=datetime.utcnow)

    job = relationship("TrainingJob", back_populates="metrics")


# ─────────────────────────────────────────────────────────────────
# Resource Snapshots
# ─────────────────────────────────────────────────────────────────

class ResourceSnapshot(Base):
    __tablename__ = "resource_snapshots"

    id          = Column(Integer, primary_key=True, index=True)
    job_id      = Column(Integer, ForeignKey("training_jobs.id"), nullable=True)
    cpu_percent = Column(Float, default=0.0)
    ram_gb      = Column(Float, default=0.0)
    gpu_percent = Column(Float, default=0.0)
    vram_gb     = Column(Float, default=0.0)
    disk_gb     = Column(Float, default=0.0)
    recorded_at = Column(DateTime, default=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────
# Checkpoints
# ─────────────────────────────────────────────────────────────────

class Checkpoint(Base):
    __tablename__ = "checkpoints"

    id           = Column(Integer, primary_key=True, index=True)
    job_id       = Column(Integer, ForeignKey("training_jobs.id"), nullable=False)
    epoch        = Column(Integer, nullable=False)
    file_path    = Column(String(512), nullable=False)
    val_loss     = Column(Float, nullable=True)
    accuracy     = Column(Float, nullable=True)
    is_verified  = Column(Boolean, default=False)
    created_at   = Column(DateTime, default=datetime.utcnow)

    job = relationship("TrainingJob", back_populates="checkpoints")


# ─────────────────────────────────────────────────────────────────
# Recovery Attempts
# ─────────────────────────────────────────────────────────────────

class RecoveryAttempt(Base):
    __tablename__ = "recovery_attempts"

    id             = Column(Integer, primary_key=True, index=True)
    job_id         = Column(Integer, ForeignKey("training_jobs.id"), nullable=False)
    attempt_number = Column(Integer, default=1)
    error_type     = Column(String(100), default="unknown")
    error_message  = Column(Text, default="")
    action_taken   = Column(String(255), default="")
    status         = Column(SAEnum(RecoveryStatus), default=RecoveryStatus.PENDING)
    diagnosis      = Column(Text, default="")
    outcome_notes  = Column(Text, default="")
    created_at     = Column(DateTime, default=datetime.utcnow)
    resolved_at    = Column(DateTime, nullable=True)

    job = relationship("TrainingJob", back_populates="recoveries")


# ─────────────────────────────────────────────────────────────────
# Experiments
# ─────────────────────────────────────────────────────────────────

class Experiment(Base):
    __tablename__ = "experiments"

    id             = Column(Integer, primary_key=True, index=True)
    job_id         = Column(Integer, ForeignKey("training_jobs.id"), nullable=True, unique=True)
    name           = Column(String(255), nullable=False)
    description    = Column(Text, default="")
    framework      = Column(String(50), default="pytorch")
    model_type     = Column(String(100), default="")
    hyperparams    = Column(JSON, default=dict)
    dataset_ref    = Column(String(512), default="")
    best_val_loss  = Column(Float, nullable=True)
    best_accuracy  = Column(Float, nullable=True)
    final_metrics  = Column(JSON, default=dict)
    status         = Column(String(50), default="pending")
    tags           = Column(JSON, default=list)
    created_at     = Column(DateTime, default=datetime.utcnow)
    completed_at   = Column(DateTime, nullable=True)

    job             = relationship("TrainingJob", back_populates="experiment")
    eval_result     = relationship("EvaluationResult", back_populates="experiment", uselist=False)
    recommendations = relationship("Recommendation", back_populates="experiment")


# ─────────────────────────────────────────────────────────────────
# Evaluation Results
# ─────────────────────────────────────────────────────────────────

class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id             = Column(Integer, primary_key=True, index=True)
    experiment_id  = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    dataset_split  = Column(String(20), default="validation")  # validation | test
    metrics        = Column(JSON, default=dict)
    report_path    = Column(String(512), default="")
    created_at     = Column(DateTime, default=datetime.utcnow)

    experiment = relationship("Experiment", back_populates="eval_result")


# ─────────────────────────────────────────────────────────────────
# AI Advisor Recommendations
# ─────────────────────────────────────────────────────────────────

class Recommendation(Base):
    __tablename__ = "recommendations"

    id              = Column(Integer, primary_key=True, index=True)
    experiment_id   = Column(Integer, ForeignKey("experiments.id"), nullable=True)
    title           = Column(String(255), nullable=False)
    description     = Column(Text, default="")
    rationale       = Column(Text, default="")
    proposed_config = Column(JSON, default=dict)
    status          = Column(String(50), default="pending")   # pending | approved | rejected | applied
    source          = Column(String(50), default="rules")     # rules | llm | trend
    priority        = Column(Integer, default=0)
    created_at      = Column(DateTime, default=datetime.utcnow)
    resolved_at     = Column(DateTime, nullable=True)

    experiment = relationship("Experiment", back_populates="recommendations")


# ─────────────────────────────────────────────────────────────────
# Notifications
# ─────────────────────────────────────────────────────────────────

class NotificationLog(Base):
    __tablename__ = "notification_logs"

    id         = Column(Integer, primary_key=True, index=True)
    level      = Column(SAEnum(NotificationLevel), default=NotificationLevel.INFO)
    title      = Column(String(255), nullable=False)
    message    = Column(Text, default="")
    channel    = Column(String(50), default="desktop")     # desktop | email
    sent       = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
