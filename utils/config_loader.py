"""
Configuration loader utility.
Reads config.yaml and merges with environment-variable overrides.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load .env if it exists (silently skip if absent)
load_dotenv(dotenv_path=Path(".env"), override=False)

_cached_config: dict[str, Any] | None = None


def load_config(path: Path | str = "config.yaml") -> dict[str, Any]:
    """
    Load YAML configuration from *path*.
    Results are cached after the first call.

    Environment variables with the prefix ``ATS__`` override nested keys:
      ATS__LLM__ENABLED=true  →  config["llm"]["enabled"] = True
    """
    global _cached_config
    if _cached_config is not None:
        return _cached_config

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh) or {}

    # Apply environment-variable overrides (ATS__SECTION__KEY=value)
    prefix = "ATS__"
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        parts = env_key[len(prefix):].lower().split("__")
        node = config
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        # Attempt type coercion
        leaf = parts[-1]
        if env_val.lower() in ("true", "false"):
            node[leaf] = env_val.lower() == "true"
        elif env_val.isdigit():
            node[leaf] = int(env_val)
        else:
            try:
                node[leaf] = float(env_val)
            except ValueError:
                node[leaf] = env_val

    _cached_config = config
    return config


def reset_config_cache() -> None:
    """Force reload on next call (useful in tests)."""
    global _cached_config
    _cached_config = None
