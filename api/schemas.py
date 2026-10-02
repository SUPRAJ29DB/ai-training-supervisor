
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Job Schemas ───────────────────────────────────────────────────────────────

class JobCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str = ""
    framework: str = "pytorch"
    model_type: str = "classifier"
    config: dict[str, Any] = Field(default_factory=dict)
    priority: int = 0


class JobResponse(BaseModel):
    id: int
    name: str
    description: str
    framework: str
    model_type: str
    status: str
    config: dict[str, Any]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: str
    current_epoch: int
    total_epochs: int

    model_config = {"from_attributes": True}


class JobStatusUpdate(BaseModel):
    action: str   # start | pause | resume | stop


# ── Metric Schemas ────────────────────────────────────────────────────────────

class MetricResponse(BaseModel):
    id: int
    job_id: int
    epoch: int
    train_loss: Optional[float] = None
    val_loss:   Optional[float] = None
    accuracy:   Optional[float] = None
    recorded_at: datetime

    model_config = {"from_attributes": True}


# ── Experiment Schemas ────────────────────────────────────────────────────────

class ExperimentCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str = ""
    framework: str = "pytorch"
    model_type: str = ""
    hyperparams: dict[str, Any] = Field(default_factory=dict)
    dataset_ref: str = ""
    tags: list[str] = Field(default_factory=list)


class ExperimentResponse(BaseModel):
    id: int
    name: str
    description: str
    framework: str
    model_type: str
    hyperparams: dict[str, Any]
    dataset_ref: str
    best_val_loss:  Optional[float] = None
    best_accuracy:  Optional[float] = None
    final_metrics:  dict[str, Any]
    status: str
    tags: list[str]
    created_at: datetime
    completed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


# ── Report Schemas ────────────────────────────────────────────────────────────

class EvaluationResponse(BaseModel):
    id: int
    experiment_id: int
    dataset_split: str
    metrics: dict[str, Any]
    report_path: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Recommendation Schemas ────────────────────────────────────────────────────

class RecommendationResponse(BaseModel):
    id: int
    title: str
    description: str
    rationale: str
    proposed_config: dict[str, Any]
    status: str
    source: str
    priority: int
    created_at: datetime

    model_config = {"from_attributes": True}


class RecommendationStatusUpdate(BaseModel):
    status: str   # approved | rejected


# ── Resource / Health Schemas ─────────────────────────────────────────────────

class ResourceSnapshotResponse(BaseModel):
    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    gpu_utilization: float
    vram_used_gb: float
    vram_total_gb: float
    disk_used_gb: float
    disk_total_gb: float
