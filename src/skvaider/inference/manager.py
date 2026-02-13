import asyncio
import functools
import hashlib
import json
import re
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, Concatenate, Literal, ParamSpec, Protocol, TypeVar

import anyio
import httpx
import psutil
import structlog

from skvaider.inference import metrics
from skvaider.inference.config import ModelConfig
from skvaider.utils import TaskManager, slugify

log = structlog.get_logger()

P = ParamSpec("P")
R = TypeVar("R")


class HasLock(Protocol):
    _lock: asyncio.Lock


SelfT = TypeVar("SelfT", bound=HasLock)


class UserManagerLock:
    def __init__(self):
        self._users = 0
        self._manager_lock = asyncio.Lock()
        self._user_lock = asyncio.Lock()

    async def user_acquire(self):
        async with self._user_lock:
            if self._users == 0:
                # We can only start using this lock if the manager isn't running.
                await self._manager_lock.acquire()
            self._users += 1

    async def user_release(self):
        async with self._user_lock:
            self._users -= 1
            if self._users == 0:
                # The manager now may start again.
                self._manager_lock.release()

    async def manager_acquire(self):
        await self._manager_lock.acquire()

    def manager_release(self):
        self._manager_lock.release()


def locked(
    func: Callable[Concatenate[SelfT, P], Coroutine[Any, Any, R]],
) -> Callable[Concatenate[SelfT, P], Coroutine[Any, Any, R]]:
    """Decorator that acquires self._lock before executing an async method."""

    @functools.wraps(func)
    async def wrapper(self: SelfT, *args: P.args, **kwargs: P.kwargs) -> R:
        async with self._lock:  # pyright: ignore[reportPrivateUsage]
            return await func(self, *args, **kwargs)

    return wrapper


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
            if not model.process:
                continue
            try:
                proc = psutil.Process(model.process.pid)
                usage = proc.memory_info().rss
                for child in proc.children(recursive=True):
                    try:
                        usage += child.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                current = self._model_usage.get(model.config.id, 0)
                self._model_usage[model.config.id] = max(current, usage)
            except psutil.NoSuchProcess:
                pass

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

        # Update Prometheus metrics
        metrics.inference_vram_bytes.labels(backend=self.id, type="total").set(
            self.total
        )
        metrics.inference_vram_bytes.labels(backend=self.id, type="used").set(
            self.used
        )
        metrics.inference_vram_bytes.labels(backend=self.id, type="free").set(
            self.free
        )

        log.info(
            f"{self.id} memory total={self.total:,} used={self.used:,} free={self.free:,}"
        )

    async def update_model_usage(self) -> None:
        """Update VRAM usage for all models with a single rocm-smi call."""
        pid_to_model: dict[int, str] = {}
        for model in self._manager.list_models():
            if not model.process:
                continue
            pid_to_model[model.process.pid] = model.config.id
            try:
                proc = psutil.Process(model.process.pid)
                for child in proc.children(recursive=True):
                    pid_to_model[child.pid] = model.config.id
            except psutil.NoSuchProcess:
                pass

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


class Model:
    process: asyncio.subprocess.Process | None = None
    endpoint: str | None = None
    config: ModelConfig
    datadir: Path
    process_status: Literal["stopped", "running", "starting", "stopping"] = (
        "stopped"
    )
    health_status: Literal["healthy", "unhealthy", ""] = ""
    health_check_interval: float = 300  # every 5 minutes
    health_check_timeout: float = (
        600  # ten minutes for now ... XXX we might want to poll /health more frequently and only do this if no requests are coming in, otherwise we get blocked.
    )
    verification_data: dict[str, list[float]] | None = None

    file_size: int = 0

    _port_found: asyncio.Event
    _tasks: TaskManager
    _host = "127.0.0.1"
    status_changed: asyncio.Event
    lock: UserManagerLock

    async def _check_health(self) -> bool: ...

    def __init__(self, config: ModelConfig):
        self.config = config
        self._port_found = asyncio.Event()
        self._tasks = TaskManager()
        self.status_changed = asyncio.Event()
        self.lock = UserManagerLock()

        if self.is_embedding:
            self._check_health = self._check_embedding_health
        else:
            self._check_health = self._check_completion_health

    def _notify_status_changed(self) -> None:
        """Notify watchers that status changed. Set and immediately clear."""
        self.status_changed.set()
        self.status_changed.clear()

    @property
    def status(self) -> set[str]:
        """Return a set of status flags for this model.

        This is modelled after Ceph's markers for placement group health to allow
        different levels of granularity to support different use cases.

        """
        result: set[str] = set()
        result.add(self.process_status)
        result.add(self.health_status)

        if set(["running", "healthy"]) <= result:
            result.add("active")
        else:
            result.add("inactive")

        # Some status might be an empty string. Filter that out.
        result = result - set([""])

        return result

    @property
    def is_embedding(self) -> bool:
        return bool(
            set(self.config.cmd_args) & set(["--embedding", "--embeddings"])
        )

    @property
    def slug(self) -> str:
        assert self.config.id is not None
        slug = slugify(self.config.id, 64)
        # The hash as suffix to assist auto-completion in shells
        slug += "-" + self.config.files[0].hash[:8]
        # if we have multiple hashes, add hash of hashes
        if len(self.config.files) > 1:
            hash_of_hashes = hashlib.sha256()
            for f in self.config.files:
                hash_of_hashes.update(bytes.fromhex(f.hash))
            slug += f"-{hash_of_hashes.hexdigest()[:8]}"
        return slug

    def url_to_filename(self, url: str) -> Path:
        # parse the url to get the last path component
        parsed = httpx.URL(url)
        path = parsed.path
        last_component = path.split("/")[-1]
        return self.datadir / last_component

    @property
    def model_files(self) -> list[Path]:
        return [self.url_to_filename(file.url) for file in self.config.files]

    def update_filesize(self) -> None:
        """Set initial memory estimate from model file sizes."""
        total = 0
        for f in self.model_files:
            if f.exists():
                total += f.stat().st_size
        self.file_size = total

    @property
    def integrity_marker_file(self) -> Path:
        return self.datadir / "integrity.ok"

    async def download(self) -> None:
        assert self.datadir
        if self.integrity_marker_file.exists() and all(
            f.exists() for f in self.model_files
        ):
            log.info(
                f"{self.slug}: found valid cached data, no download needed."
            )
            self.update_filesize()
            return

        verify_got_hashes: list[str] = []
        for model_file in self.config.files:
            url = model_file.url
            log.info(f"{self.slug}: Downloading {url} ...")
            if (model_file := self.url_to_filename(url)).exists():
                model_file.unlink()
            got_hash_ = hashlib.sha256()
            async with httpx.AsyncClient(timeout=30) as client:
                async with client.stream(
                    "GET", url, follow_redirects=True
                ) as response:
                    download_size = int(
                        response.headers.get("content-length", 0)
                    )
                    download_status = 0
                    response.raise_for_status()

                    async def log_progress():
                        while True:
                            await asyncio.sleep(5)
                            if download_size:
                                progress = int(
                                    (download_status / download_size) * 100
                                )
                                log.info(f"{self.slug}: {progress}%")
                            else:
                                log.info(
                                    f"{self.slug}: {download_status:,d} (unknown size)"
                                )

                    task = self._tasks.create(log_progress)
                    try:
                        async with await anyio.open_file(model_file, "ab") as f:
                            async for chunk in response.aiter_bytes():
                                download_status += len(chunk)
                                got_hash_.update(chunk)
                                await f.write(chunk)
                    finally:
                        task.cancel()

            got_hash = got_hash_.hexdigest()
            verify_got_hashes.append(got_hash)

        for expected, got in zip(
            [file.hash for file in self.config.files], verify_got_hashes
        ):
            if expected != got:
                raise ValueError(got)

        self.integrity_marker_file.touch()
        self.update_filesize()
        log.info(f"{self.slug}: success")

    async def start(self) -> None:
        """Start the model process."""
        log.info("Starting model", model=self.config.id)
        assert self.config.id is not None
        assert self.process is None
        assert self.process_status == "stopped"

        start_time = time.time()
        self.process_status = "starting"
        llama_server = self.config.llama_server
        if len(llama_server.parts) == 1:
            resolved = shutil.which(str(llama_server))
            assert resolved
            llama_server = Path(resolved)
        # fmt: off
        cmd: list[str] = [
                str(llama_server),
                # Keep the alias as first argument to make ps output
                # easier to read.
                "-a", self.config.id,

                # Model
                "--model", str(self.model_files[0]),
                "--jinja", # XXX allow/require control per model?
                "--ctx-size", str(self.config.context_size),

                # Network
                "--host", self._host,
                "--port", "0",  # let the kernel select a free port

                # Security
                "--no-webui",

                # Monitoring
                "--metrics",
                "--slots",
            ]
        cmd += self.config.cmd_args
        # fmt: on
        log.debug("cli", argv=" ".join(cmd))
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            self._tasks.create(self._monitor_process)
            self._tasks.create(
                self._monitor_output, args=[self.process.stderr, True]
            )
            self._tasks.create(
                self._monitor_output, args=[self.process.stdout, False]
            )
            startup_task = self._tasks.create(self._wait_for_startup)
            await startup_task
            self._tasks.create(self._monitor_health)
        except (Exception, asyncio.CancelledError):
            await self.terminate()
            raise
        log.info("Model started", model=self.config.id, endpoint=self.endpoint)
        self.process_status = "running"
        self._notify_status_changed()

        # Update metrics
        duration = time.time() - start_time
        metrics.inference_model_load_duration_seconds.labels(
            model=self.config.id
        ).observe(duration)
        metrics.inference_model_status.labels(model=self.config.id).set(1)

    async def terminate(self) -> None:
        """Terminate the process, escalating to kill if necessary."""
        log.info("Terminating model", model=self.config.id)
        self.process_status = "stopping"

        self._tasks.terminate()

        if self.process:
            pid = self.process.pid
            log.info(
                "Terminating model process",
                model=self.config.id,
                pid=pid,
            )
            try:
                self.process.terminate()
            except ProcessLookupError:
                log.exception("error terminating process", pid=pid)
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                log.info(
                    "Killing unresponsive model process",
                    model=self.config.id,
                )
                try:
                    self.process.kill()
                except ProcessLookupError:
                    pass
                await self.process.wait()
            log.info("process terminated", model=self.config.id, pid=pid)
        log.info("resetting model state", model=self.config.id)
        self._port_found.clear()
        self.process = None
        self.endpoint = None
        self.process_status = "stopped"

        # Update metrics
        metrics.inference_model_status.labels(model=self.config.id).set(0)
        self.health_status = ""
        self._notify_status_changed()

    async def _monitor_process(self) -> None:
        """Monitor whether our process has exited."""
        assert self.process is not None
        await self.process.wait()
        if self.process_status in ["stopped", "stopping"]:
            return
        log.error(
            "Process exited unexpectedly",
            model=self.config.id,
            returncode=self.process.returncode,
        )
        # Clean up
        await self.terminate()

    async def _monitor_output(
        self,
        stream: asyncio.StreamReader | None,
        is_stderr: bool,
    ) -> None:
        stream_name = "stderr" if is_stderr else "stdout"
        if stream is None:
            log.warning(f"No stream for {stream_name}.")
            return
        while True:
            line = await stream.readline()
            if not line:
                break
            line_str = line.decode("utf-8", errors="replace").strip()
            if not line_str:
                continue
            log.debug(
                "llama-server",
                model=self.config.id,
                stream=stream_name,
                line=line_str,
            )
            if is_stderr and self.endpoint is None:
                match = re.search(
                    r"main: server is listening on http://[^:]+:(\d+)",
                    line_str,
                )
                if match:
                    port = int(match.group(1))
                    self.endpoint = f"http://{self._host}:{port}"
                    self._port_found.set()

    async def _check_embedding_health(self) -> bool:
        async with httpx.AsyncClient(
            timeout=self.health_check_timeout
        ) as client:
            input_texts = ["health check"]
            expected_embeddings: list[list[float]] | None = None
            if self.verification_data:
                # input texts are keys
                input_texts = list(self.verification_data.keys())
                expected_embeddings = list(self.verification_data.values())

            resp = await client.post(
                f"{self.endpoint}/v1/embeddings",
                json={
                    "input": input_texts,
                    "model": self.config.id,
                },
            )

            if resp.status_code != 200:
                log.warning(
                    "Health check failed",
                    model=self.config.id,
                    status=resp.status_code,
                )
                return False

            if expected_embeddings is not None:
                data = resp.json()
                for i, item in enumerate(data.get("data", [])):
                    embedding = item.get("embedding", [])
                    if not embedding:
                        log.warning(
                            "Health check failed: missing embedding",
                            model=self.config.id,
                        )
                        return False
                    # Compare with expected
                    expected = expected_embeddings[i]
                    if len(embedding) != len(expected):
                        log.warning(
                            "Health check failed: embedding size mismatch",
                            model=self.config.id,
                        )
                        return False
                    # Allow small numerical differences
                    for a, b in zip(embedding, expected):
                        if abs(a - b) > 1e-2:
                            log.warning(
                                "Health check failed: embedding value mismatch",
                                model=self.config.id,
                                index=i,
                                expected=a,
                                got=b,
                                input_text=input_texts[i],
                            )
                            return False
            return True

    async def _check_completion_health(self) -> bool:
        async with httpx.AsyncClient(
            timeout=self.health_check_timeout
        ) as client:
            resp = await client.post(
                f"{self.endpoint}/v1/completions",
                json={"prompt": "2+2=", "max_tokens": 8, "n": 1},
            )
            if resp.status_code != 200:
                log.warning(
                    "Health check failed",
                    model=self.config.id,
                    status=resp.status_code,
                )
                return False
            return True

    async def _monitor_health(self) -> None:
        """Periodically check if the model is responsive."""
        new_status = ""
        interval = 0
        while True:
            # This is a bit complicated to handle varying intervals during warmup
            # and manage the event trigger
            # 1. Status update and sleep
            changed = new_status != self.health_status
            self.health_status = new_status
            if changed:
                self._notify_status_changed()
            await asyncio.sleep(interval)

            # 2. Fast loop without an actual health check to wait until we see the
            #    model starting or running and the endpoint exposed.
            if set(["starting", "running"]) & self.status and not self.endpoint:
                new_status = ""
                interval = 0.1
                continue

            # 3. Real health check immediately afterwards and switching to a slower interval.
            try:
                new_status = (
                    "healthy" if (await self._check_health()) else "unhealthy"
                )
            except Exception as e:
                log.exception(
                    "Health check error", model=self.config.id, error=str(e)
                )
                new_status = "unhealthy"

            interval = self.health_check_interval

    async def _wait_for_startup(self) -> None:
        await self._port_found.wait()
        async with httpx.AsyncClient() as client:
            while True:
                try:
                    resp = await client.get(f"{self.endpoint}/health")
                    if resp.status_code == 200:
                        return
                except Exception:
                    pass
                await asyncio.sleep(0.5)


class Manager:
    models_dir: Path
    models: dict[str, Model]
    monitors: dict[str, MemoryMonitor]
    _tasks: list[asyncio.Task[None]]

    def __init__(self, models_dir: Path):
        self.tasks = TaskManager()
        self.models_dir = models_dir
        self.models = {}
        self._lock = asyncio.Lock()

        self.monitors = {"ram": RAMMonitor(self)}
        if shutil.which("rocm-smi"):
            self.monitors["rocm"] = ROCmMemoryMonitor(self)

        for monitor in self.monitors.values():
            self.tasks.poll(monitor.update_global_usage, interval=10)
            self.tasks.poll(monitor.update_model_usage, interval=10)

    def add_model(self, model: Model) -> None:
        assert model.config.id is not None
        assert model.config.id not in self.models
        self.models[model.config.id] = model
        model.datadir = self.models_dir / model.slug
        model.datadir.mkdir(exist_ok=True)

    def list_models(self) -> list[Model]:
        return list(self.models.values())

    @locked
    async def start_model(
        self,
        model_name: str,
        timeout: int = 120,  # XXX the timeout might need to be model specific? and might need to be communicated to the gateway?
    ) -> Model:
        model = self.models[model_name]
        await model.lock.manager_acquire()
        try:
            if "active" not in model.status:
                try:
                    await asyncio.wait_for(model.start(), timeout=timeout)
                except asyncio.TimeoutError:
                    log.error("Timeout starting model", model=model_name)
                    await model.terminate()
                    raise
                # XXX let this trigger on the monitor via the event handler?
                for monitor in self.monitors.values():
                    await monitor.update_global_usage()
        finally:
            model.lock.manager_release()
        return model

    async def use_model(
        self,
        model_name: str,
    ) -> Model | None:
        model = self.models.get(model_name)
        if not model:
            return
        if "active" not in model.status:
            return
        return model

    @locked
    async def unload_model(self, model_name: str) -> None:
        model = self.models[model_name]
        await model.lock.manager_acquire()
        try:
            if model.process_status in ["running", "starting"]:  # idempotent
                await model.terminate()
                # XXX let this trigger on the monitor via the event handler?
                for monitor in self.monitors.values():
                    await monitor.update_global_usage()
        finally:
            model.lock.manager_release()

    async def shutdown(self) -> None:
        self.tasks.terminate()
        for model in list(self.models.values()):
            await model.lock.manager_acquire()
            try:
                await model.terminate()
            finally:
                model.lock.manager_release()

        # It would be cleaner if we'd use a lock here, but shutdown otherwise
        # can end up locked infinitely it seems.
        self.models.clear()
