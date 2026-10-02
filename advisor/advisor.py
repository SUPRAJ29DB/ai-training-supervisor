"""
AI Advisor – analyses experiment results and generates evidence-based
improvement recommendations.

IMPORTANT: The advisor NEVER claims that a recommendation WILL improve
performance. It proposes experiments with a stated rationale based on
observed metric patterns.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from advisor.experiment_planner import ExperimentPlanner
from advisor.trend_analyzer import TrendAnalyzer
from database.db import get_session
from database.repositories import (
    MetricsRepository, ExperimentRepository, RecommendationRepository,
)
from supervisor.event_bus import EventType, bus

logger = logging.getLogger(__name__)


class Advisor:
    """
    Improvement recommendation engine.

    Usage
    -----
    advisor = Advisor(config=config.get("llm"))
    advisor.analyse_experiment(experiment_id=1)
    """

    def __init__(self, llm_config: Optional[dict[str, Any]] = None) -> None:
        self._llm_cfg  = llm_config or {}
        self._analyser = TrendAnalyzer()
        self._planner  = ExperimentPlanner()

    def analyse_experiment(self, experiment_id: int) -> list[dict[str, Any]]:
        """
        Analyse metrics for *experiment_id* and store recommendations.
        Returns the list of recommendation dicts.
        """
        db = get_session()
        try:
            exp_repo    = ExperimentRepository(db)
            metrics_repo = MetricsRepository(db)
            rec_repo    = RecommendationRepository(db)

            experiment = exp_repo.get(experiment_id)
            if not experiment:
                logger.error("Experiment %d not found.", experiment_id)
                return []

            job_id = experiment.job_id
            raw_metrics = metrics_repo.get_for_job(job_id) if job_id else []
            metrics_dicts = [
                {
                    "epoch":      m.epoch,
                    "train_loss": m.train_loss,
                    "val_loss":   m.val_loss,
                    "accuracy":   m.accuracy,
                }
                for m in raw_metrics
            ]

            # ── Analyse trends ────────────────────────────────────
            analysis = self._analyser.analyse(metrics_dicts)
            findings = analysis.get("findings", [])

            if not findings:
                logger.info("No actionable findings for experiment %d.", experiment_id)
                return []

            # ── Generate candidate experiments ────────────────────
            base_config = experiment.hyperparams or {}
            candidates = self._planner.generate_candidates(base_config, findings)

            # ── Optionally enrich with LLM ─────────────────────────
            llm_recs: list[dict] = []
            if self._llm_cfg.get("enabled"):
                llm_recs = self._query_llm(experiment, metrics_dicts, findings)

            # ── Build recommendations ─────────────────────────────
            recommendations: list[dict[str, Any]] = []

            for finding in findings:
                rec = rec_repo.create(
                    title=f"Finding: {finding[:80]}",
                    description=finding,
                    rationale="Based on observed training metric patterns.",
                    experiment_id=experiment_id,
                    source="trend",
                    status="pending",
                )
                recommendations.append({"id": rec.id, "title": rec.title})

            for cand in candidates:
                rationale = cand.pop("_rationale", "Candidate experiment from trend analysis.")
                rec = rec_repo.create(
                    title=f"Proposed experiment: {rationale[:80]}",
                    description=rationale,
                    rationale="Evidence: " + ", ".join(findings),
                    proposed_config=cand,
                    experiment_id=experiment_id,
                    source="rules",
                    status="pending",
                )
                recommendations.append({"id": rec.id, "title": rec.title})

            for lr in llm_recs:
                rec = rec_repo.create(
                    title=lr.get("title", "LLM recommendation"),
                    description=lr.get("description", ""),
                    rationale=lr.get("rationale", ""),
                    proposed_config=lr.get("proposed_config", {}),
                    experiment_id=experiment_id,
                    source="llm",
                    status="pending",
                )
                recommendations.append({"id": rec.id, "title": rec.title})

            bus.emit_simple(
                EventType.RECOMMENDATION_READY,
                source="advisor",
                experiment_id=experiment_id,
                count=len(recommendations),
            )
            return recommendations
        finally:
            db.close()

    def _query_llm(self, experiment, metrics_dicts, findings) -> list[dict]:
        try:
            from integrations.llm.ollama_client import OllamaClient
            from advisor.prompt_templates import build_recommendation_prompt

            client = OllamaClient(
                host=self._llm_cfg.get("host", "localhost"),
                port=self._llm_cfg.get("port", 11434),
                model=self._llm_cfg.get("model", "mistral"),
                timeout=self._llm_cfg.get("timeout_seconds", 30),
            )
            prompt = build_recommendation_prompt(
                experiment.hyperparams or {},
                experiment.final_metrics or {},
                findings,
            )
            raw = client.generate(prompt)
            # Parse JSON block from response
            import re
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                data = json.loads(m.group())
                return data.get("recommendations", [])
        except Exception as exc:
            logger.warning("LLM advisor query failed: %s", exc)
        return []
