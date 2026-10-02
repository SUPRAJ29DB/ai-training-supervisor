"""
Prediction Runner – generates predictions on a dataset using a loaded model.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def run_pytorch_predictions(model: Any, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    import torch
    model.eval()
    tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        if outputs.ndim > 1 and outputs.shape[1] > 1:
            preds = torch.argmax(outputs, dim=1)
        else:
            preds = (torch.sigmoid(outputs) > 0.5).long().squeeze()
    return preds.cpu().numpy()


def run_sklearn_predictions(model: Any, X: np.ndarray) -> np.ndarray:
    return model.predict(X)


def run_predictions(
    model: Any,
    X: np.ndarray,
    framework: str = "pytorch",
    device: str = "cpu",
) -> np.ndarray:
    """Generate predictions. Never uses test data during training/tuning."""
    if framework == "pytorch":
        return run_pytorch_predictions(model, X, device)
    elif framework == "sklearn":
        return run_sklearn_predictions(model, X)
    else:
        logger.warning("Unknown framework '%s' – trying sklearn interface.", framework)
        return run_sklearn_predictions(model, X)
