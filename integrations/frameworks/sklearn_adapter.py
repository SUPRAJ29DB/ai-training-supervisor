"""
Scikit-learn Framework Adapter.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

from integrations.frameworks.base import BaseAdapter

logger = logging.getLogger(__name__)


class SklearnAdapter(BaseAdapter):
    def get_framework_name(self) -> str:
        return "sklearn"

    def train(self, config: dict[str, Any]) -> None:
        raise NotImplementedError("Use the standalone train_sklearn.py script.")

    def save_checkpoint(self, path: str, epoch: int) -> None:
        pass  # Sklearn models are pickled at end of training

    def load_checkpoint(self, path: str) -> Any:
        with open(path, "rb") as fh:
            return pickle.load(fh)

    def evaluate(self, config: dict[str, Any]) -> dict[str, Any]:
        return {}
