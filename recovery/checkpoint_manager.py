"""
Checkpoint Manager – saves, verifies, prunes, and restores model checkpoints.
Works with PyTorch checkpoints (.pt / .pth). Extendable to other frameworks.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from database.db import get_session
from database.repositories import CheckpointRepository

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages checkpoint files on disk and metadata in the database.

    Parameters
    ----------
    base_dir:
        Root checkpoint directory (e.g., artifacts/checkpoints).
    max_per_job:
        Maximum number of checkpoints to keep per job.
    """

    def __init__(self, base_dir: str = "artifacts/checkpoints", max_per_job: int = 5) -> None:
        self._base_dir = Path(base_dir)
        self._max_per_job = max_per_job
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint_dir(self, job_id: int) -> Path:
        d = self._base_dir / f"job_{job_id}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(
        self,
        job_id: int,
        epoch: int,
        state_dict: Any,
        val_loss: Optional[float] = None,
        accuracy: Optional[float] = None,
        framework: str = "pytorch",
    ) -> Path:
        """
        Save a checkpoint to disk and record it in the database.
        Returns the checkpoint file path.
        """
        ckpt_dir = self.checkpoint_dir(job_id)
        filename = f"epoch_{epoch:04d}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pt"
        filepath = ckpt_dir / filename

        if framework == "pytorch":
            try:
                import torch
                torch.save(state_dict, filepath)
                logger.info("Checkpoint saved: %s", filepath)
            except Exception as exc:
                logger.error("Failed to save checkpoint: %s", exc)
                raise
        else:
            # For other frameworks, expect state_dict to be a bytes-like object
            with open(filepath, "wb") as fh:
                fh.write(state_dict)

        # Record in DB
        db = get_session()
        try:
            repo = CheckpointRepository(db)
            ckpt = repo.save(
                job_id=job_id,
                epoch=epoch,
                file_path=str(filepath),
                val_loss=val_loss,
                accuracy=accuracy,
                is_verified=False,
            )
            # Verify immediately (file exists and is readable)
            if filepath.exists() and filepath.stat().st_size > 0:
                repo.mark_verified(ckpt.id)
                logger.debug("Checkpoint id=%d verified.", ckpt.id)

            # Prune old checkpoints
            self._prune(job_id, repo)
        finally:
            db.close()

        return filepath

    def _prune(self, job_id: int, repo: CheckpointRepository) -> None:
        checkpoints = repo.get_all_for_job(job_id)
        if len(checkpoints) <= self._max_per_job:
            return
        # Delete oldest (first in epoch-ascending order)
        to_delete = checkpoints[: len(checkpoints) - self._max_per_job]
        for ckpt in to_delete:
            p = Path(ckpt.file_path)
            if p.exists():
                p.unlink()
                logger.info("Pruned checkpoint %s", p)
            repo.delete(ckpt.id)

    def get_latest_verified(self, job_id: int) -> Optional[str]:
        """Return the file path of the latest verified checkpoint or None."""
        db = get_session()
        try:
            repo = CheckpointRepository(db)
            ckpt = repo.get_latest_verified(job_id)
            return ckpt.file_path if ckpt else None
        finally:
            db.close()

    def restore(self, job_id: int, framework: str = "pytorch") -> Optional[Any]:
        """
        Load and return the latest verified checkpoint state_dict.
        Returns None if no verified checkpoint exists.
        """
        path_str = self.get_latest_verified(job_id)
        if not path_str:
            logger.warning("No verified checkpoint for job %d.", job_id)
            return None
        path = Path(path_str)
        if not path.exists():
            logger.error("Checkpoint file missing: %s", path)
            return None

        if framework == "pytorch":
            import torch
            state = torch.load(path, map_location="cpu")
            logger.info("Checkpoint restored from %s", path)
            return state
        else:
            with open(path, "rb") as fh:
                return fh.read()

    def list_checkpoints(self, job_id: int) -> list[dict]:
        db = get_session()
        try:
            repo = CheckpointRepository(db)
            return [
                {
                    "id":          c.id,
                    "epoch":       c.epoch,
                    "file_path":   c.file_path,
                    "val_loss":    c.val_loss,
                    "accuracy":    c.accuracy,
                    "is_verified": c.is_verified,
                    "created_at":  c.created_at.isoformat(),
                }
                for c in repo.get_all_for_job(job_id)
            ]
        finally:
            db.close()
