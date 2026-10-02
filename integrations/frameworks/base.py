"""
Base Framework Adapter – defines the common interface that all framework
adapters must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAdapter(ABC):
    """
    Abstract base for framework adapters.
    All training scripts accessed through this adapter must implement
    these methods to integrate with the supervisor.
    """

    @abstractmethod
    def get_framework_name(self) -> str:
        """Return the framework name (e.g., 'pytorch')."""

    @abstractmethod
    def train(self, config: dict[str, Any]) -> None:
        """
        Launch training with the given configuration.
        Must emit metric updates to metrics.jsonl in the working directory.
        """

    @abstractmethod
    def save_checkpoint(self, path: str, epoch: int) -> None:
        """Save a checkpoint to the given path."""

    @abstractmethod
    def load_checkpoint(self, path: str) -> Any:
        """Load and return a checkpoint."""

    @abstractmethod
    def evaluate(self, config: dict[str, Any]) -> dict[str, Any]:
        """Evaluate the model and return a metrics dict."""
