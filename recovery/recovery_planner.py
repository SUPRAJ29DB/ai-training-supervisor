"""
Recovery Planner – selects the best safe recovery action for a classified error.
Only actions on the policy allowlist may be selected.
"""

from __future__ import annotations

import logging
from typing import Any

from recovery.error_classifier import ClassifiedError, ErrorType
from supervisor.policy_engine import PolicyEngine

logger = logging.getLogger(__name__)


class RecoveryPlan:
    def __init__(self, action: str, params: dict[str, Any], rationale: str) -> None:
        self.action   = action
        self.params   = params
        self.rationale = rationale

    def __repr__(self) -> str:
        return f"RecoveryPlan(action={self.action}, params={self.params})"


class RecoveryPlanner:
    """
    Chooses the single best recovery action from the classified error's
    suggested_actions list, filtered through the policy engine.
    """

    def __init__(self, policy: PolicyEngine) -> None:
        self._policy = policy

    def plan(
        self,
        classified: ClassifiedError,
        job_config: dict[str, Any],
        attempt_number: int,
    ) -> RecoveryPlan:
        """
        Return the best allowed RecoveryPlan.
        Falls back to 'stop_job' if no other action is permitted.
        """
        for action in classified.suggested_actions:
            if not self._policy.is_action_allowed(action):
                continue
            if self._policy.requires_approval(action):
                logger.info("Action '%s' requires approval – skipping for auto-recovery.", action)
                continue
            params = self._build_params(action, job_config, attempt_number)
            rationale = f"Selected '{action}' for error type '{classified.error_type.value}'."
            return RecoveryPlan(action=action, params=params, rationale=rationale)

        # Final fallback
        return RecoveryPlan(
            action="stop_job",
            params={},
            rationale="No auto-recoverable action available. Stopping safely.",
        )

    def _build_params(
        self,
        action: str,
        job_config: dict[str, Any],
        attempt: int,
    ) -> dict[str, Any]:
        if action == "reduce_batch_size":
            current = job_config.get("batch_size", 32)
            # Reduce by 25% per attempt, minimum 4
            new_batch = max(4, int(current * (0.75 ** attempt)))
            return {"batch_size": new_batch}
        return {}
