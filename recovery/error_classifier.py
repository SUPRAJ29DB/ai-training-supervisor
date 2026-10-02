"""
Error Classifier – categorises training errors into known types.
Uses regex-based rule matching first, then falls back to keyword heuristics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ErrorType(str, Enum):
    OOM            = "out_of_memory"
    INVALID_CONFIG = "invalid_configuration"
    MISSING_FILE   = "missing_file"
    NAN_LOSS       = "nan_loss"
    RUNTIME        = "runtime_error"
    STALL          = "job_stalled"
    UNKNOWN        = "unknown"


@dataclass
class ClassifiedError:
    error_type: ErrorType
    severity: str              # low | medium | high | critical
    raw_message: str
    suggested_actions: list[str]


# Pattern → (ErrorType, severity, suggested_actions)
_RULES: list[tuple[re.Pattern, ErrorType, str, list[str]]] = [
    (
        re.compile(r"cuda out of memory|out of memory|oom", re.IGNORECASE),
        ErrorType.OOM, "high",
        ["reduce_batch_size", "clear_gpu_cache"],
    ),
    (
        re.compile(r"filenotfounderror|no such file", re.IGNORECASE),
        ErrorType.MISSING_FILE, "high",
        ["stop_job"],
    ),
    (
        re.compile(r"nan loss detected|loss is nan|loss=nan", re.IGNORECASE),
        ErrorType.NAN_LOSS, "medium",
        ["resume_from_checkpoint", "restart_job"],
    ),
    (
        re.compile(r"invalid.*config|configuration.*error|keyerror|valueerror.*config", re.IGNORECASE),
        ErrorType.INVALID_CONFIG, "medium",
        ["stop_job"],
    ),
    (
        re.compile(r"runtimeerror|cuda error|device-side assert", re.IGNORECASE),
        ErrorType.RUNTIME, "high",
        ["resume_from_checkpoint", "restart_job"],
    ),
]


class ErrorClassifier:
    """
    Classifies a raw error string into a structured ClassifiedError.
    """

    def classify(self, error_message: str) -> ClassifiedError:
        for pattern, etype, severity, actions in _RULES:
            if pattern.search(error_message):
                return ClassifiedError(
                    error_type=etype,
                    severity=severity,
                    raw_message=error_message,
                    suggested_actions=actions,
                )
        # Default: unknown
        return ClassifiedError(
            error_type=ErrorType.UNKNOWN,
            severity="critical",
            raw_message=error_message,
            suggested_actions=["stop_job"],
        )

    def classify_stall(self, reason: str) -> ClassifiedError:
        return ClassifiedError(
            error_type=ErrorType.STALL,
            severity="medium",
            raw_message=reason,
            suggested_actions=["resume_from_checkpoint", "restart_job"],
        )
