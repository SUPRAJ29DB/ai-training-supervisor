"""
Report Generator – creates JSON and text evaluation reports.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ReportGenerator:
    def __init__(self, reports_dir: str = "artifacts/reports") -> None:
        self._dir = Path(reports_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        experiment_id: int,
        job_id: int,
        metrics: dict[str, Any],
        dataset_split: str = "validation",
        overfitting_info: dict | None = None,
        extra: dict | None = None,
    ) -> str:
        """
        Write a JSON evaluation report and return its file path.
        """
        report = {
            "experiment_id": experiment_id,
            "job_id":        job_id,
            "dataset_split": dataset_split,
            "metrics":       metrics,
            "overfitting":   overfitting_info or {},
            "extra":         extra or {},
            "generated_at":  datetime.utcnow().isoformat(),
        }
        filename = f"eval_exp{experiment_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        path = self._dir / filename
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        logger.info("Evaluation report saved: %s", path)
        return str(path)

    def load(self, report_path: str) -> dict[str, Any]:
        with open(report_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
