"""
Dataset Validator – runs integrity checks on training datasets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    passed: bool = True
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def add_issue(self, msg: str) -> None:
        self.issues.append(msg)
        self.passed = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


class DatasetValidator:
    """
    Validates a dataset before training begins.
    Supports numpy arrays directly or a path to a CSV / directory.
    """

    def validate_arrays(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str = "classification",
    ) -> ValidationReport:
        report = ValidationReport()
        n_samples, *_ = X.shape

        # ── Basic checks ──────────────────────────────────────────
        if n_samples == 0:
            report.add_issue("Dataset is empty (0 samples).")
            return report

        if len(y) != n_samples:
            report.add_issue(
                f"X has {n_samples} samples but y has {len(y)} – mismatch."
            )

        # ── NaN / Inf checks ──────────────────────────────────────
        nan_count = int(np.isnan(X).sum())
        inf_count = int(np.isinf(X).sum())
        if nan_count > 0:
            report.add_issue(f"X contains {nan_count} NaN values.")
        if inf_count > 0:
            report.add_issue(f"X contains {inf_count} Inf values.")

        # ── Label checks (classification) ─────────────────────────
        if task_type == "classification":
            classes, counts = np.unique(y, return_counts=True)
            min_count = int(counts.min())
            if min_count < 2:
                report.add_warning(
                    f"Class {classes[counts.argmin()]} has only {min_count} sample(s) – "
                    "consider resampling."
                )
            report.stats["n_classes"] = int(len(classes))
            report.stats["class_distribution"] = {
                str(c): int(cnt) for c, cnt in zip(classes, counts)
            }

        report.stats.update({
            "n_samples":  n_samples,
            "n_features": X.shape[1] if X.ndim > 1 else 1,
            "task_type":  task_type,
        })

        return report

    def validate_path(self, path: str) -> ValidationReport:
        report = ValidationReport()
        p = Path(path)
        if not p.exists():
            report.add_issue(f"Dataset path does not exist: {path}")
            return report
        if p.is_file():
            if p.stat().st_size == 0:
                report.add_issue(f"Dataset file is empty: {path}")
            else:
                report.stats["file_size_mb"] = round(p.stat().st_size / 1024 / 1024, 2)
        elif p.is_dir():
            files = list(p.rglob("*"))
            report.stats["n_files"] = len(files)
            if not files:
                report.add_issue(f"Dataset directory is empty: {path}")
        return report
