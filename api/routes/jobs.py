"""
Job management API routes.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.schemas import JobCreateRequest, JobResponse, JobStatusUpdate
from database.db import get_db
from database.models import JobStatus
from database.repositories import JobRepository

logger = logging.getLogger(__name__)
router = APIRouter()


def get_job_or_404(job_id: int, db: Session) -> object:
    job = JobRepository(db).get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return job


@router.get("/", response_model=list[JobResponse])
def list_jobs(
    status: str | None = None,
    db: Session = Depends(get_db),
):
    """List all training jobs, optionally filtered by status."""
    js = None
    if status:
        try:
            js = JobStatus(status)
        except ValueError:
            raise HTTPException(400, f"Invalid status '{status}'.")
    return JobRepository(db).list_all(status=js)


@router.post("/", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
def create_job(request: JobCreateRequest, db: Session = Depends(get_db)):
    """Create a new training job (PENDING state)."""
    job = JobRepository(db).create(
        name=request.name,
        config=request.config,
        description=request.description,
        framework=request.framework,
        model_type=request.model_type,
        priority=request.priority,
    )
    return job


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: int, db: Session = Depends(get_db)):
    return get_job_or_404(job_id, db)


@router.post("/{job_id}/action")
def job_action(
    job_id: int,
    body: JobStatusUpdate,
    db: Session = Depends(get_db),
):
    """
    Control a job lifecycle: start | pause | resume | stop.
    Actions are forwarded to the orchestrator via the event bus.
    """
    from supervisor.event_bus import bus, EventType, Event

    get_job_or_404(job_id, db)
    action = body.action.lower()
    allowed = {"start", "pause", "resume", "stop"}
    if action not in allowed:
        raise HTTPException(400, f"Unknown action '{action}'. Allowed: {allowed}")

    # Import orchestrator singleton (set up at startup)
    try:
        from supervisor.orchestrator import Orchestrator
        # We use the event bus to decouple API from orchestrator
        event_map = {
            "start":  EventType.JOB_STARTED,
            "pause":  EventType.JOB_PAUSED,
            "resume": EventType.JOB_RESUMED,
            "stop":   EventType.JOB_STOPPED,
        }
        bus.emit_simple(event_map[action], source="api", job_id=job_id, requested=True)
    except Exception as exc:
        logger.exception("Failed to dispatch action '%s' for job %d: %s", action, job_id, exc)
        raise HTTPException(500, "Internal error dispatching action.")

    return {"job_id": job_id, "action": action, "status": "dispatched"}


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_job(job_id: int, db: Session = Depends(get_db)):
    """Delete a job (only if STOPPED or FAILED)."""
    job = get_job_or_404(job_id, db)
    if job.status not in (JobStatus.STOPPED, JobStatus.FAILED, JobStatus.COMPLETED):
        raise HTTPException(400, "Cannot delete an active job. Stop it first.")
    JobRepository(db).delete(job_id)
