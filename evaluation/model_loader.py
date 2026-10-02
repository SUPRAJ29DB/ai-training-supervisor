"""
Model Loader – framework-agnostic model loading utility.
Supports PyTorch, Scikit-learn (pickle), and custom formats.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_pytorch_model(path: str, model_class=None, device: str = "cpu") -> Any:
    """Load a PyTorch model from a .pt / .pth checkpoint."""
    import torch
    state = torch.load(path, map_location=device)
    if model_class is not None:
        model = model_class()
        model.load_state_dict(state.get("model_state_dict", state))
        model.eval()
        return model
    return state


def load_sklearn_model(path: str) -> Any:
    """Load a pickled scikit-learn model."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


def load_model(path: str, framework: str = "pytorch", **kwargs) -> Any:
    """
    Universal model loader.

    Parameters
    ----------
    path:      Path to the model file.
    framework: 'pytorch' | 'sklearn' | 'custom'.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    if framework == "pytorch":
        return load_pytorch_model(path, **kwargs)
    elif framework == "sklearn":
        return load_sklearn_model(path)
    else:
        logger.warning("Unknown framework '%s' – attempting pickle load.", framework)
        return load_sklearn_model(path)
