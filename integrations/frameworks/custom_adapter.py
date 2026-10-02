"""
Custom Framework Adapter – plug-in point for arbitrary training scripts.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from typing import Any

from integrations.frameworks.base import BaseAdapter

logger = logging.getLogger(__name__)


class CustomAdapter(BaseAdapter):
    """
    Wraps a user-supplied training script.
    The script is launched as a subprocess; it is expected to write
    metrics to metrics.jsonl in its working directory.
    """

    def __init__(self, script_path: str, working_dir: str = ".") -> None:
        self._script = script_path
        self._working_dir = working_dir

    def get_framework_name(self) -> str:
        return "custom"

    def train(self, config: dict[str, Any]) -> None:
        cmd = [sys.executable, self._script]
        subprocess.run(cmd, cwd=self._working_dir, check=True)

    def save_checkpoint(self, path: str, epoch: int) -> None:
        pass

    def load_checkpoint(self, path: str) -> Any:
        return None

    def evaluate(self, config: dict[str, Any]) -> dict[str, Any]:
        return {}
