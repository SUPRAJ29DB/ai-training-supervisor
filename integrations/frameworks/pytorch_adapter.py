"""
PyTorch Framework Adapter.
Provides a reusable training loop that emits metrics.jsonl for the supervisor.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from integrations.frameworks.base import BaseAdapter

logger = logging.getLogger(__name__)


class PyTorchAdapter(BaseAdapter):
    def get_framework_name(self) -> str:
        return "pytorch"

    def train(self, config: dict[str, Any]) -> None:
        raise NotImplementedError("Use the standalone train_pytorch.py script.")

    def save_checkpoint(self, path: str, epoch: int) -> None:
        pass  # Handled by CheckpointManager

    def load_checkpoint(self, path: str) -> Any:
        return torch.load(path, map_location="cpu")

    def evaluate(self, config: dict[str, Any]) -> dict[str, Any]:
        return {}


def write_metric(working_dir: str, epoch: int, **metrics) -> None:
    """
    Helper used by training scripts to emit a metric line.
    Appends one JSON line to <working_dir>/metrics.jsonl.
    """
    p = Path(working_dir) / "metrics.jsonl"
    record = {"epoch": epoch, **metrics}
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(
            "Using GPU: %s (%.1f GB VRAM)",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available – using CPU.")
    return device
