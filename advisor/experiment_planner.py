"""
Experiment Planner – generates candidate hyperparameter experiments based on
advisor recommendations.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Safe hyperparameter search space
_SEARCH_SPACE: dict[str, list[Any]] = {
    "learning_rate":  [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    "batch_size":     [8, 16, 32, 64, 128],
    "weight_decay":   [0.0, 1e-5, 1e-4, 1e-3],
    "dropout":        [0.0, 0.1, 0.2, 0.3, 0.5],
    "num_epochs":     [10, 20, 50, 100],
}


class ExperimentPlanner:
    """Generates candidate experiment configurations from base config + findings."""

    def generate_candidates(
        self,
        base_config: dict[str, Any],
        findings: list[str],
        max_candidates: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Return a list of candidate configs to try, based on trend findings.
        Each candidate is a copy of base_config with one parameter adjusted.
        Only safe, bounded adjustments are made.
        """
        candidates: list[dict[str, Any]] = []

        for finding in findings:
            finding_lower = finding.lower()

            if "plateau" in finding_lower:
                # Try a lower learning rate
                current_lr = base_config.get("learning_rate", 1e-3)
                new_lr = max(1e-5, current_lr * 0.3)
                candidate = copy.deepcopy(base_config)
                candidate["learning_rate"] = round(new_lr, 6)
                candidate["_rationale"] = "Reduce LR to escape plateau."
                candidates.append(candidate)

            elif "overfitting" in finding_lower:
                # Add weight decay
                candidate = copy.deepcopy(base_config)
                candidate["weight_decay"] = 1e-4
                candidate["dropout"] = min(0.5, base_config.get("dropout", 0.0) + 0.1)
                candidate["_rationale"] = "Add regularisation to combat overfitting."
                candidates.append(candidate)

            elif "accuracy is low" in finding_lower:
                # Increase epochs
                candidate = copy.deepcopy(base_config)
                candidate["num_epochs"] = min(200, base_config.get("num_epochs", 10) * 2)
                candidate["_rationale"] = "Train for more epochs to improve accuracy."
                candidates.append(candidate)

            if len(candidates) >= max_candidates:
                break

        return candidates[:max_candidates]
