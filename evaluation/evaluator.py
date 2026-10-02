"""
Evaluator – coordinates dataset validation, prediction, metrics, and reporting.
Enforces the rule: test set is ONLY used for final evaluation, never during tuning.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from database.db import get_session
from database.repositories import ExperimentRepository, EvaluationRepository
from evaluation.dataset_validator import DatasetValidator
from evaluation.metrics import compute_metrics, detect_overfitting
from evaluation.model_loader import load_model
from evaluation.prediction_runner import run_predictions
from evaluation.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Coordinates the full evaluation pipeline.

    Parameters
    ----------
    config: The ``evaluation`` section of global config.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._reports_dir = config.get("results_directory", "artifacts/reports")
        self._validator   = DatasetValidator()
        self._reporter    = ReportGenerator(self._reports_dir)

    def evaluate(
        self,
        experiment_id: int,
        job_id: int,
        model_path: str,
        X_val: np.ndarray,
        y_val: np.ndarray,
        framework: str = "pytorch",
        task_type: str = "classification",
        dataset_split: str = "validation",
        train_losses: Optional[list[float]] = None,
        val_losses: Optional[list[float]] = None,
    ) -> dict[str, Any]:
        """
        Run evaluation on the validation (or test) set.

        IMPORTANT: Test set should only be used ONCE for final evaluation.
        Pass dataset_split='test' only after model selection is complete.
        """
        if dataset_split == "test":
            logger.warning(
                "Using TEST set for evaluation – ensure model selection is complete "
                "before calling this. Test set must not be used for tuning."
            )

        # ── Validate dataset ──────────────────────────────────────
        val_report = self._validator.validate_arrays(X_val, y_val, task_type)
        if not val_report.passed:
            logger.error("Dataset validation failed: %s", val_report.issues)
            return {"error": "Dataset validation failed", "issues": val_report.issues}

        # ── Load model ────────────────────────────────────────────
        try:
            model = load_model(model_path, framework=framework)
        except FileNotFoundError as exc:
            logger.error("Model not found: %s", exc)
            return {"error": str(exc)}

        # ── Predict ───────────────────────────────────────────────
        y_pred = run_predictions(model, X_val, framework=framework)

        # ── Compute metrics ───────────────────────────────────────
        metrics = compute_metrics(task_type, y_val, y_pred)

        # ── Overfitting detection ─────────────────────────────────
        overfitting_info = {}
        if train_losses and val_losses:
            overfitting_info = detect_overfitting(train_losses, val_losses)

        # ── Generate report ───────────────────────────────────────
        report_path = self._reporter.generate(
            experiment_id=experiment_id,
            job_id=job_id,
            metrics=metrics,
            dataset_split=dataset_split,
            overfitting_info=overfitting_info,
        )

        # ── Persist to DB ─────────────────────────────────────────
        db = get_session()
        try:
            EvaluationRepository(db).save_result(
                experiment_id=experiment_id,
                metrics=metrics,
                dataset_split=dataset_split,
                report_path=report_path,
            )
            ExperimentRepository(db).update(
                experiment_id,
                best_accuracy=metrics.get("accuracy"),
                final_metrics=metrics,
                status="evaluated",
            )
        finally:
            db.close()

        logger.info("Evaluation complete for experiment %d: %s", experiment_id, metrics)
        return {"metrics": metrics, "report_path": report_path, "overfitting": overfitting_info}
