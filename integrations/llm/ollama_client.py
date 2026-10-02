"""
Ollama Client – connects to the Raspberry Pi 5 running Ollama.

SAFETY RULES:
- This client ONLY sends text and receives text.
- Responses are NEVER directly executed.
- The Pi has NO access to the Windows machine.
- LLM is completely optional; all callers must handle ConnectionError gracefully.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    HTTP client for the Ollama REST API (v1).

    Parameters
    ----------
    host:    IP address of the Raspberry Pi (or localhost for testing).
    port:    Ollama port (default 11434).
    model:   Model name (e.g., 'mistral', 'llama3').
    timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        host: str = "192.168.1.100",
        port: int = 11434,
        model: str = "mistral",
        timeout: int = 30,
    ) -> None:
        self._base_url = f"http://{host}:{port}"
        self._model    = model
        self._timeout  = timeout

    def is_available(self) -> bool:
        """Quick health check – returns True if the Ollama server is reachable."""
        try:
            resp = requests.get(f"{self._base_url}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, system: str = "") -> str:
        """
        Send *prompt* to Ollama and return the generated text.

        Parameters
        ----------
        prompt: The user prompt.
        system: Optional system message.
        """
        payload: dict[str, Any] = {
            "model":  self._model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system

        try:
            resp = requests.post(
                f"{self._base_url}/api/generate",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Ollama server at {self._base_url} is not reachable. "
                "Check that the Raspberry Pi is on the same network and Ollama is running."
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Ollama request timed out after {self._timeout}s.")
        except Exception as exc:
            logger.error("Ollama request failed: %s", exc)
            raise
