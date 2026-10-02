"""
Resource Monitor – collects CPU, RAM, GPU, and disk metrics using psutil / py3nvml.
Runs in a background thread and emits RESOURCE_ALERT events when limits are exceeded.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import psutil

from supervisor.event_bus import EventType, bus

logger = logging.getLogger(__name__)

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False
    logger.warning("py3nvml / NVML not available – GPU metrics disabled.")


@dataclass
class ResourceSnapshot:
    cpu_percent: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    gpu_utilization: float = 0.0     # 0-100
    vram_used_gb: float = 0.0
    vram_total_gb: float = 0.0
    disk_used_gb: float = 0.0
    disk_total_gb: float = 0.0


def _get_gpu_metrics() -> tuple[float, float, float]:
    """Returns (gpu_util%, vram_used_gb, vram_total_gb) or zeros if NVML unavailable."""
    if not _NVML_AVAILABLE:
        return 0.0, 0.0, 0.0
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem  = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return (
            float(util.gpu),
            mem.used / (1024 ** 3),
            mem.total / (1024 ** 3),
        )
    except Exception as exc:
        logger.debug("GPU metric read error: %s", exc)
        return 0.0, 0.0, 0.0


def collect_snapshot() -> ResourceSnapshot:
    """Collect a single resource snapshot synchronously."""
    cpu  = psutil.cpu_percent(interval=0.2)
    ram  = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    gpu_util, vram_used, vram_total = _get_gpu_metrics()
    return ResourceSnapshot(
        cpu_percent=cpu,
        ram_used_gb=ram.used / (1024 ** 3),
        ram_total_gb=ram.total / (1024 ** 3),
        gpu_utilization=gpu_util,
        vram_used_gb=vram_used,
        vram_total_gb=vram_total,
        disk_used_gb=disk.used / (1024 ** 3),
        disk_total_gb=disk.total / (1024 ** 3),
    )


class ResourceMonitor:
    """
    Background thread that periodically samples system resources.
    Emits RESOURCE_ALERT events when configured thresholds are exceeded.
    """

    def __init__(self, config: dict[str, Any], interval_seconds: float = 10.0) -> None:
        self._config = config.get("resources", {})
        self._interval = interval_seconds
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest: Optional[ResourceSnapshot] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="resource-monitor"
        )
        self._thread.start()
        logger.info("Resource monitor started (interval=%.1fs).", self._interval)

    def stop(self) -> None:
        self._running = False

    @property
    def latest(self) -> Optional[ResourceSnapshot]:
        with self._lock:
            return self._latest

    def _loop(self) -> None:
        while self._running:
            try:
                snap = collect_snapshot()
                with self._lock:
                    self._latest = snap
                self._check_thresholds(snap)
            except Exception as exc:
                logger.exception("Resource monitor error: %s", exc)
            time.sleep(self._interval)

    def _check_thresholds(self, snap: ResourceSnapshot) -> None:
        alerts: list[str] = []

        max_cpu = self._config.get("max_cpu_percent", 90)
        if snap.cpu_percent > max_cpu:
            alerts.append(f"CPU {snap.cpu_percent:.1f}% > {max_cpu}%")

        max_ram = self._config.get("max_ram_gb", 12)
        if snap.ram_used_gb > max_ram:
            alerts.append(f"RAM {snap.ram_used_gb:.1f} GB > {max_ram} GB")

        max_vram_frac = self._config.get("max_gpu_memory_fraction", 0.90)
        if snap.vram_total_gb > 0:
            vram_frac = snap.vram_used_gb / snap.vram_total_gb
            if vram_frac > max_vram_frac:
                alerts.append(
                    f"VRAM {snap.vram_used_gb:.1f}/{snap.vram_total_gb:.1f} GB "
                    f"({vram_frac*100:.0f}% > {max_vram_frac*100:.0f}%)"
                )

        max_disk = self._config.get("max_disk_gb", 50)
        if snap.disk_used_gb > max_disk:
            alerts.append(f"Disk {snap.disk_used_gb:.1f} GB > {max_disk} GB")

        if alerts:
            logger.warning("Resource alerts: %s", "; ".join(alerts))
            bus.emit_simple(
                EventType.RESOURCE_ALERT,
                source="resource_monitor",
                alerts=alerts,
                snapshot={
                    "cpu_percent":   snap.cpu_percent,
                    "ram_used_gb":   snap.ram_used_gb,
                    "gpu_util":      snap.gpu_utilization,
                    "vram_used_gb":  snap.vram_used_gb,
                    "disk_used_gb":  snap.disk_used_gb,
                },
            )
