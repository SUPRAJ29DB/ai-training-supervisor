"""
Diagnosis Agent – analyses classified errors and produces structured diagnoses.
Uses rule-based analysis first; optionally queries the LLM (Raspberry Pi / Ollama).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from recovery.error_classifier import ClassifiedError, ErrorType

logger = logging.getLogger(__name__)


class DiagnosisAgent:
    """
    Produces human-readable diagnoses and recovery recommendations from
    classified errors. Optionally enriches diagnoses via LLM.

    Parameters
    ----------
    llm_config:
        The ``llm`` section of the global config.
        If ``enabled`` is False or the LLM is unreachable, rule-based
        diagnosis is used.
    """

    def __init__(self, llm_config: Optional[dict[str, Any]] = None) -> None:
        self._llm_cfg = llm_config or {}
        self._llm_enabled = self._llm_cfg.get("enabled", False)

    def diagnose(
        self,
        classified: ClassifiedError,
        job_config: dict[str, Any],
        recent_metrics: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Return a diagnosis dict with keys:
          - summary    : str
          - root_cause : str
          - actions    : list[str]
          - source     : 'rules' | 'llm'
        """
        rule_diagnosis = self._rule_based(classified, job_config, recent_metrics)

        if self._llm_enabled:
            try:
                llm_enrichment = self._query_llm(classified, job_config, recent_metrics)
                rule_diagnosis["summary"]    = llm_enrichment.get("summary", rule_diagnosis["summary"])
                rule_diagnosis["root_cause"] = llm_enrichment.get("root_cause", rule_diagnosis["root_cause"])
                rule_diagnosis["source"]     = "llm"
            except Exception as exc:
                logger.warning("LLM diagnosis unavailable, using rules. Error: %s", exc)

        return rule_diagnosis

    # ── Rule-based diagnosis ──────────────────────────────────────────────────

    def _rule_based(
        self,
        classified: ClassifiedError,
        job_config: dict[str, Any],
        recent_metrics: list[dict[str, Any]],
    ) -> dict[str, Any]:
        etype = classified.error_type
        batch = job_config.get("batch_size", "unknown")
        lr    = job_config.get("learning_rate", "unknown")

        summaries = {
            ErrorType.OOM: (
                f"GPU ran out of memory. Current batch_size={batch}. "
                "Reducing batch size or enabling gradient checkpointing may help."
            ),
            ErrorType.MISSING_FILE: (
                "A required file was not found. Check dataset paths and script arguments."
            ),
            ErrorType.NAN_LOSS: (
                f"Training loss became NaN. This is often caused by a learning rate ({lr}) "
                "that is too high, exploding gradients, or corrupt input data."
            ),
            ErrorType.INVALID_CONFIG: (
                "A configuration error was detected. Review hyperparameters and dataset references."
            ),
            ErrorType.RUNTIME: (
                "A runtime error occurred. Check the job log for the full traceback."
            ),
            ErrorType.STALL: (
                "Training appears to have stalled – no progress recorded within the timeout."
            ),
            ErrorType.UNKNOWN: (
                "An unrecognised error occurred. Manual inspection of the logs is required."
            ),
        }

        return {
            "summary":    summaries.get(etype, classified.raw_message),
            "root_cause": etype.value,
            "actions":    classified.suggested_actions,
            "source":     "rules",
        }

    # ── LLM diagnosis (optional) ──────────────────────────────────────────────

    def _query_llm(
        self,
        classified: ClassifiedError,
        job_config: dict[str, Any],
        recent_metrics: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Send error context to the Ollama LLM on the Raspberry Pi.
        Returns a dict with 'summary' and 'root_cause' keys.

        SAFETY: The LLM response is NEVER directly executed.
        Only structured text fields are extracted.
        """
        from integrations.llm.ollama_client import OllamaClient
        from advisor.prompt_templates import build_diagnosis_prompt

        client = OllamaClient(
            host=self._llm_cfg.get("host", "localhost"),
            port=self._llm_cfg.get("port", 11434),
            model=self._llm_cfg.get("model", "mistral"),
            timeout=self._llm_cfg.get("timeout_seconds", 30),
        )
        prompt = build_diagnosis_prompt(classified, job_config, recent_metrics)
        response = client.generate(prompt)
        return self._parse_llm_response(response)

    def _parse_llm_response(self, raw: str) -> dict[str, Any]:
        """Extract structured fields from LLM text. Never executes code."""
        import re
        summary_match    = re.search(r"SUMMARY:\s*(.+?)(?:ROOT_CAUSE:|$)", raw, re.DOTALL | re.IGNORECASE)
        root_cause_match = re.search(r"ROOT_CAUSE:\s*(.+?)(?:$)", raw, re.DOTALL | re.IGNORECASE)
        return {
            "summary":    (summary_match.group(1).strip()    if summary_match    else raw[:200]),
            "root_cause": (root_cause_match.group(1).strip() if root_cause_match else "unknown"),
        }
