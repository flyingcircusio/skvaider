import asyncio
import csv
import io
import json
import subprocess
from abc import ABC, abstractmethod

import psutil
import structlog
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from .manager import Manager
    from .model import Model


from skvaider.inference import metrics

log = structlog.get_logger()


class MemoryMonitor(ABC):
    """Abstract base class for monitoring memory usage (RAM or VRAM)."""

    id: str

    total: int = 0
    used: int = 0
    free: int = 0

    _model_usage: dict[str, int]
    _manager: "Manager"

    def __init__(self, manager: "Manager"):
        self._manager = manager
        self._model_usage = {}

    @abstractmethod
    async def update_global_usage(self) -> None:
        """Update global memory statistics (total, used, free) in bytes."""
        ...

    @abstractmethod
    async def update_model_usage(self) -> None:
        """Update memory usage for all models by collecting PIDs and querying."""
        ...

    def model_usage(self, model: "Model") -> int:
        return self._model_usage.get(model.config.id, 0)

    def _update_vram_metrics(self) -> None:
        """Update Prometheus VRAM metrics (for GPU monitors)."""
        metrics.inference_vram_bytes.labels(backend=self.id, type="total").set(
            self.total
        )
        metrics.inference_vram_bytes.labels(backend=self.id, type="used").set(
            self.used
        )
        metrics.inference_vram_bytes.labels(backend=self.id, type="free").set(
            self.free
        )

    def _update_model_metrics(self) -> None:
        """Update Prometheus per-model memory metrics."""
        for model_id, usage in self._model_usage.items():
            metrics.inference_memory_bytes.labels(
                model=model_id, type="model"
            ).set(usage)


class RAMMonitor(MemoryMonitor):
    """Memory monitor using psutil for system RAM."""

    id = "ram"

    async def update_global_usage(self) -> None:
        mem = psutil.virtual_memory()
        self.total = mem.total
        self.used = mem.used
        self.free = mem.available

        # Update Prometheus metrics
        metrics.inference_memory_bytes.labels(model="", type="total").set(
            self.total
        )
        metrics.inference_memory_bytes.labels(model="", type="used").set(
            self.used
        )
        metrics.inference_memory_bytes.labels(model="", type="free").set(
            self.free
        )

        log.info(
            f"{self.id} memory total={self.total:,} used={self.used:,} free={self.free:,}"
        )

    async def update_model_usage(self) -> None:
        for model in self._manager.list_models():
            pids = await model.pids()
            if not pids:
                continue
            usage = 0
            for pid in pids:
                try:
                    usage += psutil.Process(pid).memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            current = self._model_usage.get(model.config.id, 0)
            self._model_usage[model.config.id] = max(current, usage)

            # Update Prometheus metrics
            metrics.inference_memory_bytes.labels(
                model=model.config.id, type="model"
            ).set(self._model_usage[model.config.id])


class ROCmMemoryMonitor(MemoryMonitor):
    """Memory monitor using rocm-smi for AMD GPU VRAM."""

    id = "rocm"

    async def update_global_usage(self) -> None:
        """Update ROCm card VRAM memory statistics."""
        proc = await asyncio.create_subprocess_exec(
            "rocm-smi",
            "--json",
            "--showmeminfo",
            "all",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=5
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise
        assert proc.returncode is not None

        if proc.returncode != 0:
            raise subprocess.CalledProcessError(
                proc.returncode,
                "rocm-smi",
                stdout,
                stderr,
            )

        # Instead of giving us a JSON with no data, it gives us an empty stdout and a warning on stderr
        if not stdout:
            log.debug(
                "rocm-smi returned no data",
                stderr=stderr.decode("utf-8", errors="replace"),
            )
            return

        data = json.loads(stdout.decode("utf-8"))

        # Sum VRAM across all cards
        total = 0
        used = 0
        for key, card_data in data.items():
            if not key.startswith("card"):
                continue
            total += int(card_data["VRAM Total Memory (B)"])
            used += int(card_data["VRAM Total Used Memory (B)"])

        self.total = total
        self.used = used
        self.free = total - used

        self._update_vram_metrics()

        log.info(
            f"{self.id} memory total={self.total:,} used={self.used:,} free={self.free:,}"
        )

    async def update_model_usage(self) -> None:
        """Update VRAM usage for all models with a single rocm-smi call."""
        pid_to_model: dict[int, str] = {}
        for model in self._manager.list_models():
            for pid in await model.pids():
                pid_to_model[pid] = model.config.id

        if not pid_to_model:
            return

        proc = await asyncio.create_subprocess_exec(
            "rocm-smi",
            "--json",
            "--showpids",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=5
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return
        assert proc.returncode is not None

        if proc.returncode != 0:
            log.warning(
                "rocm-smi --showpids failed",
                returncode=proc.returncode,
                stderr=stderr.decode("utf-8", errors="replace"),
            )
            return

        if not stdout:
            # Don't log. This happens regularly when there is no active ROCM usage.
            return

        data = json.loads(stdout.decode("utf-8"))
        system_data = data.get("system", {})

        # Sum up VRAM usage per model (may have multiple PIDs per model)
        model_usage: dict[str, int] = {}
        # Format: "PID123456": "process_name, gpu_count, vram_bytes, cpu_mem, unknown"
        for pid_key, value in system_data.items():
            if not pid_key.startswith("PID"):
                continue
            pid = int(pid_key[3:])
            if pid not in pid_to_model:
                continue
            model_id = pid_to_model[pid]
            parts = value.split(",")
            usage = int(parts[2].strip())
            model_usage[model_id] = model_usage.get(model_id, 0) + usage

        for model_id, usage in model_usage.items():
            current = self._model_usage.get(model_id, 0)
            self._model_usage[model_id] = max(current, usage)

        self._update_model_metrics()


class NvidiaMemoryMonitor(MemoryMonitor):
    """Memory monitor using nvidia-smi for Nvidia GPU VRAM."""

    id = "nvidia"

    async def update_global_usage(self) -> None:
        """Update Nvidia GPU VRAM memory statistics."""
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=5
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise
        assert proc.returncode is not None

        if proc.returncode != 0:
            raise subprocess.CalledProcessError(
                proc.returncode,
                "nvidia-smi",
                stdout,
                stderr,
            )

        if not stdout:
            log.debug(
                "nvidia-smi returned no data",
                stderr=stderr.decode("utf-8", errors="replace"),
            )
            return

        # Sum memory across all GPUs (one line per GPU, values in MiB)
        total = 0
        used = 0
        free = 0
        reader = csv.reader(io.StringIO(stdout.decode("utf-8")))
        for row in reader:
            if len(row) != 3:
                continue
            total += int(row[0].strip()) * 1024 * 1024  # MiB to bytes
            used += int(row[1].strip()) * 1024 * 1024
            free += int(row[2].strip()) * 1024 * 1024

        self.total = total
        self.used = used
        self.free = free

        self._update_vram_metrics()

        log.info(
            f"{self.id} memory total={self.total:,} used={self.used:,} free={self.free:,}"
        )

    async def update_model_usage(self) -> None:
        """Update VRAM usage for all models with nvidia-smi."""
        pid_to_model: dict[int, str] = {}
        for model in self._manager.list_models():
            for pid in await model.pids():
                pid_to_model[pid] = model.config.id

        if not pid_to_model:
            return

        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=5
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return
        assert proc.returncode is not None

        if proc.returncode != 0:
            log.warning(
                "nvidia-smi --query-compute-apps failed",
                returncode=proc.returncode,
                stderr=stderr.decode("utf-8", errors="replace"),
            )
            return

        if not stdout:
            return

        # Sum up VRAM usage per model (may have multiple PIDs per model)
        model_usage: dict[str, int] = {}
        reader = csv.reader(io.StringIO(stdout.decode("utf-8")))
        for row in reader:
            if len(row) != 2:
                continue
            try:
                pid = int(row[0].strip())
            except ValueError:
                log.warning(f"unexpected PID value: {row[0]}")
                continue
            if pid not in pid_to_model:
                continue
            model_id = pid_to_model[pid]
            try:
                usage = int(row[1].strip()) * 1024 * 1024  # MiB to bytes
            except ValueError:
                log.warning(f"unexpected usage value: {row[0]}")
                continue
            model_usage[model_id] = model_usage.get(model_id, 0) + usage

        for model_id, usage in model_usage.items():
            current = self._model_usage.get(model_id, 0)
            self._model_usage[model_id] = max(current, usage)

        self._update_model_metrics()
