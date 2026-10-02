"""Metrics API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.schemas import MetricResponse, ResourceSnapshotResponse
from database.db import get_db
from database.repositories import MetricsRepository
from monitoring.resource_monitor import collect_snapshot

router = APIRouter()


@router.get("/job/{job_id}", response_model=list[MetricResponse])
def get_metrics_for_job(job_id: int, db: Session = Depends(get_db)):
    """Return all training metrics for a given job."""
    return MetricsRepository(db).get_for_job(job_id)


@router.get("/resources", response_model=ResourceSnapshotResponse)
def get_current_resources():
    """Return a live system resource snapshot."""
    snap = collect_snapshot()
    return ResourceSnapshotResponse(
        cpu_percent=snap.cpu_percent,
        ram_used_gb=snap.ram_used_gb,
        ram_total_gb=snap.ram_total_gb,
        gpu_utilization=snap.gpu_utilization,
        vram_used_gb=snap.vram_used_gb,
        vram_total_gb=snap.vram_total_gb,
        disk_used_gb=snap.disk_used_gb,
        disk_total_gb=snap.disk_total_gb,
    )
