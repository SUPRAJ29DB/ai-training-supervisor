"""
Local LLM fallback – rule-based response generator used when
the Raspberry Pi / Ollama is not available.
"""

from __future__ import annotations

from recovery.error_classifier import ClassifiedError, ErrorType


class LocalLLM:
    """
    Rule-based text generator that mimics the LLM response format.
    Used as a fallback when the Ollama server is not reachable.
    """

    def generate_diagnosis(self, classified: ClassifiedError) -> str:
        templates = {
            ErrorType.OOM: (
                "SUMMARY: GPU out-of-memory error. The current batch size exceeds available VRAM.\n"
                "ROOT_CAUSE: out_of_memory"
            ),
            ErrorType.NAN_LOSS: (
                "SUMMARY: Training loss became NaN, likely due to a high learning rate or corrupt data.\n"
                "ROOT_CAUSE: nan_loss"
            ),
            ErrorType.MISSING_FILE: (
                "SUMMARY: A required file was not found. Verify dataset paths and script arguments.\n"
                "ROOT_CAUSE: missing_file"
            ),
            ErrorType.INVALID_CONFIG: (
                "SUMMARY: Invalid training configuration. Check hyperparameters and dataset settings.\n"
                "ROOT_CAUSE: invalid_configuration"
            ),
        }
        return templates.get(
            classified.error_type,
            (
                f"SUMMARY: An error of type '{classified.error_type.value}' occurred.\n"
                "ROOT_CAUSE: unknown"
            ),
        )
