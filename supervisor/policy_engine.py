"""
Safety & Policy Engine.
Enforces resource limits, approved-action allowlists, and retry caps.
"""

from __future__ import annotations

import logging
from typing import Any

from utils.logger import get_audit_logger

logger = logging.getLogger(__name__)
audit = get_audit_logger()

# ── Approved recovery actions (allowlist) ─────────────────────────────────────
APPROVED_ACTIONS: dict[str, dict[str, Any]] = {
    "reduce_batch_size": {
        "description": "Reduce batch size by a bounded factor (max 50% reduction)",
        "max_reduction_factor": 0.5,
        "requires_approval": False,
    },
    "resume_from_checkpoint": {
        "description": "Resume training from the latest verified checkpoint",
        "requires_approval": False,
    },
    "clear_gpu_cache": {
        "description": "Call torch.cuda.empty_cache()",
        "requires_approval": False,
    },
    "restart_job": {
        "description": "Stop and re-queue the job (same config)",
        "requires_approval": False,
    },
    "stop_job": {
        "description": "Safely stop the job and preserve checkpoints",
        "requires_approval": False,
    },
    "reduce_batch_size_large": {
        "description": "Reduce batch size by more than 50%",
        "requires_approval": True,
    },
    "change_architecture": {
        "description": "Modify the model architecture",
        "requires_approval": True,
    },
    "replace_dataset": {
        "description": "Replace the training dataset",
        "requires_approval": True,
    },
    "install_package": {
        "description": "Install or upgrade a Python package",
        "requires_approval": True,
    },
}


class PolicyViolationError(Exception):
    """Raised when a proposed action violates safety policy."""


class PolicyEngine:
    """
    Checks proposed actions against safety policies.

    Parameters
    ----------
    config:
        The ``safety`` section of the global config.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._cfg = config
        self._max_retries: int = config.get("max_recovery_attempts", 3)
        self._auto_stop_unknown: bool = config.get("auto_stop_on_unknown_error", True)
        self._requires_approval: list[str] = config.get("require_approval_for", [])

    # ── Action allowlist ──────────────────────────────────────────
    def is_action_allowed(self, action: str) -> bool:
        return action in APPROVED_ACTIONS

    def requires_approval(self, action: str) -> bool:
        action_meta = APPROVED_ACTIONS.get(action, {})
        return action_meta.get("requires_approval", False) or action in self._requires_approval

    def validate_action(self, action: str, job_id: int) -> None:
        """
        Raise *PolicyViolationError* if *action* is not on the allowlist.
        """
        if not self.is_action_allowed(action):
            msg = f"Action '{action}' is not on the approved action allowlist."
            audit.warning("POLICY BLOCK | job_id=%d | action=%s | reason=not_allowed", job_id, action)
            raise PolicyViolationError(msg)
        if self.requires_approval(action):
            msg = f"Action '{action}' requires explicit user approval before execution."
            audit.warning("POLICY BLOCK | job_id=%d | action=%s | reason=requires_approval", job_id, action)
            raise PolicyViolationError(msg)
        audit.info("POLICY ALLOW | job_id=%d | action=%s", job_id, action)

    # ── Retry limit ────────────────────────────────────────────────
    def check_retry_limit(self, job_id: int, current_attempts: int) -> bool:
        """Return True if another recovery attempt is allowed."""
        allowed = current_attempts < self._max_retries
        if not allowed:
            audit.warning(
                "RETRY LIMIT REACHED | job_id=%d | attempts=%d | max=%d",
                job_id, current_attempts, self._max_retries,
            )
        return allowed

    # ── Resource limits ────────────────────────────────────────────
    def check_resource_limits(self, resource_snapshot: dict[str, float]) -> list[str]:
        """
        Return a list of violation messages for resources exceeding limits.
        Empty list means all limits are satisfied.
        """
        violations: list[str] = []
        limits = {
            "cpu_percent": self._cfg.get("max_cpu_percent", 90),
            "ram_gb":       self._cfg.get("max_ram_gb", 12),
            "gpu_percent":  self._cfg.get("max_gpu_memory_fraction", 0.90) * 100,
            "disk_gb":      self._cfg.get("max_disk_gb", 50),
        }
        for key, limit in limits.items():
            value = resource_snapshot.get(key)
            if value is not None and value > limit:
                violations.append(f"{key}={value:.1f} exceeds limit={limit}")
        return violations

    # ── Unknown error policy ───────────────────────────────────────
    def should_stop_on_unknown_error(self) -> bool:
        return self._auto_stop_unknown
