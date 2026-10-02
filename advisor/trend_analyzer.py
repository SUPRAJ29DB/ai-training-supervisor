"""
Trend Analyser – analyses training metric trends to identify issues.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class TrendAnalyzer:
    def analyse(self, metrics: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Analyse a list of epoch metric dicts and return findings.
        """
        if len(metrics) < 3:
            return {"findings": [], "suggestion": "Not enough data yet."}

        train_losses = [m.get("train_loss") for m in metrics if m.get("train_loss") is not None]
        val_losses   = [m.get("val_loss")   for m in metrics if m.get("val_loss")   is not None]
        accuracies   = [m.get("accuracy")   for m in metrics if m.get("accuracy")   is not None]

        findings: list[str] = []

        # Stagnation: last 3 val_losses differ by < 0.01
        if len(val_losses) >= 3:
            recent = val_losses[-3:]
            if (max(recent) - min(recent)) < 0.01:
                findings.append("Validation loss has plateaued for the last 3 epochs.")

        # Divergence: val_loss increasing while train_loss decreasing
        if len(val_losses) >= 4 and len(train_losses) >= 4:
            vl = np.array(val_losses[-4:])
            tl = np.array(train_losses[-4:])
            if np.polyfit(range(4), vl, 1)[0] > 0 and np.polyfit(range(4), tl, 1)[0] < 0:
                findings.append(
                    "Overfitting detected: val_loss is increasing while train_loss is decreasing."
                )

        # Poor accuracy
        if accuracies and accuracies[-1] < 0.50:
            findings.append(f"Accuracy is low ({accuracies[-1]:.1%}). Model may need architectural changes.")

        return {
            "findings": findings,
            "last_train_loss": train_losses[-1] if train_losses else None,
            "last_val_loss":   val_losses[-1]   if val_losses   else None,
            "last_accuracy":   accuracies[-1]   if accuracies   else None,
        }
