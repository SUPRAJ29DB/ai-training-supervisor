"""
LLM Prompt Templates for the AI Advisor.
"""

from __future__ import annotations

from typing import Any


def build_diagnosis_prompt(classified, job_config: dict, recent_metrics: list) -> str:
    return f"""You are an AI training diagnosis assistant. Analyse the following training failure and provide a structured diagnosis.

ERROR TYPE: {classified.error_type.value}
ERROR MESSAGE: {classified.raw_message[:500]}

JOB CONFIGURATION:
- Batch size: {job_config.get('batch_size', 'unknown')}
- Learning rate: {job_config.get('learning_rate', 'unknown')}
- Framework: {job_config.get('framework', 'unknown')}

RECENT METRICS (last 5 epochs):
{_format_metrics(recent_metrics[-5:])}

Provide your response in this exact format:
SUMMARY: <one or two sentence summary of the problem>
ROOT_CAUSE: <technical root cause>

Do NOT suggest executing any code or commands. Only provide textual analysis."""


def build_recommendation_prompt(experiment_config: dict, metrics: dict, findings: list[str]) -> str:
    return f"""You are an AI training improvement advisor. Based on the following experiment results, suggest hyperparameter improvements.

EXPERIMENT CONFIGURATION:
{_format_dict(experiment_config)}

CURRENT METRICS:
{_format_dict(metrics)}

OBSERVED ISSUES:
{chr(10).join('- ' + f for f in findings)}

Provide your response as a JSON object with this structure:
{{
  "recommendations": [
    {{
      "title": "...",
      "description": "...",
      "rationale": "...",
      "proposed_config": {{"key": "value"}}
    }}
  ]
}}

Only suggest changes to: learning_rate, batch_size, weight_decay, dropout, num_epochs.
Do NOT suggest installing packages, modifying code, or executing commands."""


def _format_metrics(metrics: list) -> str:
    if not metrics:
        return "No metrics available."
    lines = []
    for m in metrics:
        lines.append(
            f"  Epoch {m.get('epoch', '?')}: "
            f"train_loss={m.get('train_loss', 'N/A')}, "
            f"val_loss={m.get('val_loss', 'N/A')}, "
            f"accuracy={m.get('accuracy', 'N/A')}"
        )
    return "\n".join(lines)


def _format_dict(d: dict) -> str:
    return "\n".join(f"  {k}: {v}" for k, v in d.items())
