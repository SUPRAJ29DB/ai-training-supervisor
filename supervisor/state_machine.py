"""
Job State Machine – enforces valid status transitions for training jobs.
"""

from __future__ import annotations

from database.models import JobStatus

# Valid transitions: {current_status: set_of_allowed_next_statuses}
TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.PENDING:    {JobStatus.QUEUED, JobStatus.STOPPED},
    JobStatus.QUEUED:     {JobStatus.RUNNING, JobStatus.STOPPED},
    JobStatus.RUNNING:    {JobStatus.PAUSED, JobStatus.COMPLETED, JobStatus.FAILED,
                           JobStatus.STOPPED, JobStatus.RECOVERING},
    JobStatus.PAUSED:     {JobStatus.RUNNING, JobStatus.STOPPED},
    JobStatus.RECOVERING: {JobStatus.RUNNING, JobStatus.FAILED, JobStatus.STOPPED},
    JobStatus.COMPLETED:  set(),          # Terminal
    JobStatus.FAILED:     {JobStatus.QUEUED},   # Allow re-queue after manual review
    JobStatus.STOPPED:    {JobStatus.QUEUED},   # Allow re-queue after manual review
}


class StateMachineError(Exception):
    """Raised when an illegal status transition is attempted."""


class JobStateMachine:
    """
    Validates and records job status transitions.
    The machine does NOT touch the database; the orchestrator is responsible
    for persisting the new status.
    """

    def __init__(self, current_status: JobStatus = JobStatus.PENDING):
        self._status = current_status

    @property
    def status(self) -> JobStatus:
        return self._status

    def can_transition(self, new_status: JobStatus) -> bool:
        return new_status in TRANSITIONS.get(self._status, set())

    def transition(self, new_status: JobStatus) -> JobStatus:
        """
        Attempt to move to *new_status*.

        Raises
        ------
        StateMachineError
            If the transition is not permitted.
        """
        if not self.can_transition(new_status):
            raise StateMachineError(
                f"Invalid transition: {self._status} → {new_status}. "
                f"Allowed: {TRANSITIONS.get(self._status, set())}"
            )
        previous = self._status
        self._status = new_status
        return previous

    @classmethod
    def allowed_transitions(cls, status: JobStatus) -> set[JobStatus]:
        return TRANSITIONS.get(status, set())
