"""
TensorFlow Framework Adapter (optional).
Only imported when TensorFlow is installed.
"""

from __future__ import annotations

import logging
from typing import Any

from integrations.frameworks.base import BaseAdapter

logger = logging.getLogger(__name__)


class TensorFlowAdapter(BaseAdapter):
    def get_framework_name(self) -> str:
        return "tensorflow"

    def train(self, config: dict[str, Any]) -> None:
        raise NotImplementedError("Use the standalone train_tensorflow.py script.")

    def save_checkpoint(self, path: str, epoch: int) -> None:
        pass

    def load_checkpoint(self, path: str) -> Any:
        try:
            import tensorflow as tf
            return tf.keras.models.load_model(path)
        except ImportError:
            logger.error("TensorFlow is not installed.")
            return None

    def evaluate(self, config: dict[str, Any]) -> dict[str, Any]:
        return {}
