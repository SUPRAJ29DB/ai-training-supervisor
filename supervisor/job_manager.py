"""
Job Manager – launches and manages training job sub-processes.
Wraps subprocess management with stdin/stdout/stderr capture,
pause/resume (via SIGSTOP/SIGCONT on Unix or DebugActiveProcess on Windows),
and graceful termination.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# On Windows, CTRL_BREAK_EVENT is used to interrupt processes in a group.
_IS_WINDOWS = sys.platform == "win32"


class RunningJob:
    """Represents a running training subprocess."""

    def __init__(
        self,
        job_id: int,
        process: subprocess.Popen,
        log_file: Path,
        on_complete: Callable[[int, int], None],
        on_error: Callable[[int, str], None],
    ) -> None:
        self.job_id = job_id
        self.process = process
        self.log_file = log_file
        self._on_complete = on_complete
        self._on_error = on_error
        self._monitor_thread = threading.Thread(
            target=self._monitor, daemon=True, name=f"monitor-job-{job_id}"
        )
        self._monitor_thread.start()

    def _monitor(self) -> None:
        """Wait for the process to finish and invoke the appropriate callback."""
        returncode = self.process.wait()
        if returncode == 0:
            logger.info("Job %d completed successfully.", self.job_id)
            self._on_complete(self.job_id, returncode)
        else:
            msg = f"Process exited with code {returncode}."
            logger.warning("Job %d failed: %s", self.job_id, msg)
            self._on_error(self.job_id, msg)

    def is_alive(self) -> bool:
        return self.process.poll() is None

    def pause(self) -> None:
        if not self.is_alive():
            return
        if _IS_WINDOWS:
            # Windows: no native SIGSTOP. We use an OS-specific approach to
            # suspend the process. Requires ctypes.
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = kernel32.OpenProcess(0x001F0FFF, False, self.process.pid)
            kernel32.SuspendThread(handle)
            logger.info("Job %d paused (Windows SuspendThread).", self.job_id)
        else:
            os.kill(self.process.pid, signal.SIGSTOP)
            logger.info("Job %d paused (SIGSTOP).", self.job_id)

    def resume(self) -> None:
        if not self.is_alive():
            return
        if _IS_WINDOWS:
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = kernel32.OpenProcess(0x001F0FFF, False, self.process.pid)
            kernel32.ResumeThread(handle)
            logger.info("Job %d resumed (Windows ResumeThread).", self.job_id)
        else:
            os.kill(self.process.pid, signal.SIGCONT)
            logger.info("Job %d resumed (SIGCONT).", self.job_id)

    def stop(self, timeout: float = 10.0) -> None:
        if not self.is_alive():
            return
        try:
            self.process.terminate()
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("Job %d did not stop gracefully; killing.", self.job_id)
            self.process.kill()
        logger.info("Job %d stopped.", self.job_id)


class JobManager:
    """
    Launches training jobs as sub-processes and keeps track of them.
    """

    def __init__(self, log_dir: Path = Path("logs")) -> None:
        self._log_dir = log_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[int, RunningJob] = {}
        self._lock = threading.Lock()

    def launch(
        self,
        job_id: int,
        script_path: str,
        args: list[str],
        working_dir: str,
        env_extra: Optional[dict[str, str]] = None,
        on_complete: Optional[Callable[[int, int], None]] = None,
        on_error: Optional[Callable[[int, str], None]] = None,
    ) -> RunningJob:
        """
        Launch a training script as a subprocess.

        Parameters
        ----------
        job_id      : Supervisor job ID.
        script_path : Path to the Python training script.
        args        : Additional CLI arguments for the script.
        working_dir : Working directory for the process.
        env_extra   : Extra environment variables to inject.
        on_complete : Called with (job_id, returncode) on success.
        on_error    : Called with (job_id, error_msg) on failure.
        """
        log_path = self._log_dir / f"job_{job_id}.log"
        env = os.environ.copy()
        if env_extra:
            env.update(env_extra)

        cmd = [sys.executable, str(script_path)] + args
        logger.info("Launching job %d: %s", job_id, " ".join(cmd))

        with open(log_path, "w", encoding="utf-8") as log_fh:
            process = subprocess.Popen(
                cmd,
                cwd=str(working_dir),
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if _IS_WINDOWS else 0,
            )

        running_job = RunningJob(
            job_id=job_id,
            process=process,
            log_file=log_path,
            on_complete=on_complete or (lambda j, r: None),
            on_error=on_error or (lambda j, e: None),
        )

        with self._lock:
            self._jobs[job_id] = running_job

        return running_job

    def get(self, job_id: int) -> Optional[RunningJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def remove(self, job_id: int) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)

    def list_active(self) -> list[int]:
        with self._lock:
            return [jid for jid, rj in self._jobs.items() if rj.is_alive()]
