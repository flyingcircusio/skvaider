import asyncio
import functools
import hashlib
import os
import shutil
import time
from abc import ABC
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import (
    Any,
    Concatenate,
    Literal,
    ParamSpec,
    Protocol,
    TypeVar,
)

import anyio
import httpx
import structlog

from skvaider.inference import metrics
from skvaider.inference.config import (
    LlamaServerModelConfig,
    ModelConfig,
    SystemdModelConfig,
    VllmModelConfig,
)
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

    def manager_locked(self):
        return self._manager_lock.locked()

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


class Model(ABC):
    config: ModelConfig
    datadir: Path
    # Set by Manager.add_model from the config's logging.log_dir.
    log_dir: Path
    verification_data: dict[str, list[float]] | None = None
    _host = "127.0.0.1"

    endpoint: str | None = None

    lock: UserManagerLock

    process: asyncio.subprocess.Process | None = None
    process_status: Literal["stopped", "running", "starting", "stopping"] = (
        "stopped"
    )
    status_changed: asyncio.Event

    health_status: Literal["healthy", "unhealthy", ""] = ""
    health_checks: dict[str, str] = {}  # check name -> "" ok, or failure reason
    health_check_interval: float = 300  # every 5 minutes
    health_check_timeout: float = 600  # ten minutes for now ... XXX we might want to poll /health more frequently and only do this if no requests are coming in, otherwise we get blocked.
    _health_checks: int = 0  # support testing

    _tasks: TaskManager

    _engine: str

    @property
    def slug(self) -> str: ...

    async def download(self) -> None: ...

    async def start(self) -> None: ...

    # Shared implementation:

    def __init__(self, config: ModelConfig):
        self.config = config
        self.lock = UserManagerLock()
        self.status_changed = asyncio.Event()
        self._tasks = TaskManager()

        if self.is_embedding():
            self._check_health = self._check_embedding_health
        else:
            self._check_health = self._check_completion_health

    def is_embedding(self) -> bool:
        return self.config.task == "embedding"

    def _notify_status_changed(self) -> None:
        """Notify watchers that status changed. Set and immediately clear."""
        self.status_changed.set()
        self.status_changed.clear()

    @property
    def integrity_marker_file(self) -> Path:
        return self.datadir / "integrity.ok"

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

    async def _check_health(self) -> dict[str, str]: ...

    async def _monitor_health(self) -> None:
        """Periodically check if the model is responsive."""
        new_status = ""
        interval = 0
        while True:
            self._health_checks += 1
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
            if not self.endpoint:
                new_status = ""
                interval = 0.1
                continue

            # 3. Real health check immediately afterwards and switching to a slower interval.
            try:
                new_checks = await self._check_health()
            except Exception as e:
                log.exception("Health check error", model=self.config.id)
                new_checks = {"health": str(e) or type(e).__name__}

            self.health_checks = new_checks
            new_status = "unhealthy" if any(new_checks.values()) else "healthy"
            interval = self.health_check_interval

    async def _wait_for_startup(self) -> None:
        expected_endpoint = f"http://{self._host}:{self.config.port}"
        async with httpx.AsyncClient() as client:
            while True:
                try:
                    log.info(
                        "checking endpoint",
                        endpoint=f"{expected_endpoint}/health",
                    )
                    resp = await client.get(f"{expected_endpoint}/health")
                    if resp.status_code == 200:
                        self.endpoint = expected_endpoint
                        return
                except Exception:
                    pass
                await asyncio.sleep(0.5)

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
            status=self.process_status,
        )
        # Clean up
        await self.terminate()

    async def _check_embedding_health(self) -> dict[str, str]:
        checks: dict[str, str] = {}
        async with httpx.AsyncClient(
            timeout=self.health_check_timeout
        ) as client:
            input_texts = ["health check"]
            expected_embeddings: list[list[float]] | None = None
            if self.verification_data:
                input_texts = list(self.verification_data.keys())
                expected_embeddings = list(self.verification_data.values())

            resp = await client.post(
                f"{self.endpoint}/v1/embeddings",
                json={"input": input_texts, "model": self.config.id},
            )
            if resp.status_code != 200:
                checks["embedding"] = f"HTTP {resp.status_code}"
                return checks

            checks["embedding"] = ""

            if expected_embeddings is not None:
                data = resp.json()
                for i, item in enumerate(data.get("data", [])):
                    embedding = item.get("embedding", [])
                    if not embedding:
                        checks["numerical"] = "missing embedding"
                        return checks
                    expected = expected_embeddings[i]
                    if len(embedding) != len(expected):
                        checks["numerical"] = (
                            f"size mismatch: got {len(embedding)}, expected {len(expected)}"
                        )
                        return checks
                    for j, (a, b) in enumerate(zip(embedding, expected)):
                        if abs(a - b) > 1e-2:
                            checks["numerical"] = (
                                f"value mismatch at dim {j} for {input_texts[i]!r}: got {a:.6f}, expected {b:.6f}"
                            )
                            return checks
                checks["numerical"] = ""
        return checks

    async def _check_completion_health(self) -> dict[str, str]:
        async with httpx.AsyncClient(
            timeout=self.health_check_timeout
        ) as client:
            resp = await client.post(
                f"{self.endpoint}/v1/completions",
                json={"prompt": "2+2=", "max_tokens": 8, "n": 1},
            )
            return {
                "completion": ""
                if resp.status_code == 200
                else f"HTTP {resp.status_code}"
            }

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
        self.process = None
        self.endpoint = None
        self.process_status = "stopped"

        # Update metrics
        metrics.inference_model_status.labels(model=self.config.id).set(0)
        self.health_status = ""
        self._notify_status_changed()

    async def _launch_process(
        self, cmd: list[str], extra_env: dict[str, str] | None = None
    ) -> None:
        """Start the subprocess and wait for /health.

        stdout/stderr of the child go directly to inference-<id>.log in
        self.log_dir so each model has its own file.
        """
        log.debug("cli", argv=" ".join(cmd))
        log_path = self.log_dir / f"inference-{self.config.id}.log"
        log.info(
            "Logging model output to file",
            model=self.config.id,
            log_path=str(log_path),
        )
        log_file = open(log_path, "a")
        process_env = {**os.environ, **(extra_env or {})}
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=log_file,
            stderr=log_file,
            env=process_env,
            # XXX we may need to consider improving our termination *if* the cleanup
            # of a new session should be unreliable.
            start_new_session=True,
        )
        log_file.close()  # child inherited the FD; we no longer need our copy
        try:
            self._tasks.create(self._monitor_process)
            startup_task = self._tasks.create(self._wait_for_startup)
            await startup_task
            self._tasks.create(self._monitor_health)
        except (Exception, asyncio.CancelledError):
            await self.terminate()
            raise


class LlamaModel(Model):
    _engine = "llama-server"

    def __init__(self, config: LlamaServerModelConfig):
        super().__init__(config)
        self._config = config  # Allow access to type-specific config

    def is_embedding(self) -> bool:
        return self.config.task == "embedding"

    @property
    def slug(self) -> str:
        assert self.config.id is not None
        slug = slugify(self.config.id, 64)
        # The hash as suffix to assist auto-completion in shells
        slug += "-" + self._config.files[0].hash[:8]
        # if we have multiple hashes, add hash of hashes
        if len(self._config.files) > 1:
            hash_of_hashes = hashlib.sha256()
            for f in self._config.files:
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
        return [self.url_to_filename(file.url) for file in self._config.files]

    async def download(self) -> None:
        assert self.datadir
        if self.integrity_marker_file.exists() and all(
            f.exists() for f in self.model_files
        ):
            log.info(
                f"{self.slug}: found valid cached data, no download needed."
            )
            return

        verify_got_hashes: list[str] = []
        for model_file in self._config.files:
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
            [file.hash for file in self._config.files], verify_got_hashes
        ):
            if expected != got:
                raise ValueError(got)

        self.integrity_marker_file.touch()
        log.info(f"{self.slug}: success")

    async def start(self) -> None:
        """Start the model process."""
        log.info("Starting model", model=self.config.id)
        assert self.config.id is not None
        assert self.process is None
        assert self.process_status == "stopped"

        start_time = time.time()
        self.process_status = "starting"
        llama_server = self._config.llama_server
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
                "--ctx-size", str(self._config.context_size),

                # Network
                "--host", self._host,
                "--port", str(self.config.port),

                # Security
                "--no-webui",

                # Monitoring
                "--metrics",
                "--slots",
            ]
        if self.config.task == "embedding" and "--embeddings" not in self._config.cmd_args:
            cmd += ["--embeddings"]
        cmd += self._config.cmd_args
        # fmt: on
        await self._launch_process(cmd)
        log.info("Model started", model=self.config.id, endpoint=self.endpoint)
        self.process_status = "running"
        self._notify_status_changed()

        # Update metrics
        duration = time.time() - start_time
        metrics.inference_model_load_duration_seconds.labels(
            model=self.config.id
        ).observe(duration)
        metrics.inference_model_status.labels(model=self.config.id).set(1)


class VllmModel(Model):
    _engine = "vllm"

    def __init__(self, config: VllmModelConfig):
        super().__init__(config)
        self._config = config

    def is_embedding(self) -> bool:
        return self.config.task == "embedding"

    @property
    def slug(self) -> str:
        assert self.config.id is not None
        slug = slugify(self.config.id, 64)
        # The hash as suffix to assist auto-completion in shells
        slug += "-" + self._config.revision
        return slug

    async def start(self) -> None:
        """Start the model process."""
        log.info("Starting model", model=self.config.id)
        assert self.config.id is not None
        assert self.process is None
        assert self.process_status == "stopped"

        start_time = time.time()
        self.process_status = "starting"
        executable = self._config.vllm
        if len(executable.parts) == 1:
            resolved = shutil.which(str(executable))
            assert resolved
            executable = Path(resolved)
        # fmt: off
        cmd: list[str] = [
                str(executable),
                "serve",
                # Keep the alias as first argument to make ps output
                # easier to read.
                "--served-model-name", str(self.config.id),

                # https://docs.vllm.ai/en/latest/cli/serve

                # Frontend
                "--host", self._host,
                "--port", str(self.config.port),

                # ModelConfig
                str(self._config.repo),
                "--revision", str(self._config.revision),
                "--no-trust-remote-code",
                "--max-model-len", str(self._config.context_size),

                # LoadConfig
                "--download-dir", str(self.datadir),

                # CacheConfig
                "--gpu-memory-utilization", "0.95",

                # SchedulerConfig
                "--async-scheduling",

                # Security
                "--disable-fastapi-docs",

                # Monitoring / Metrics
                # XXX
            ]
        cmd += self._config.cmd_args
        # fmt: on
        await self._launch_process(cmd, extra_env=self._config.env or None)
        log.info("Model started", model=self.config.id, endpoint=self.endpoint)
        self.process_status = "running"
        self._notify_status_changed()

        # Update metrics
        duration = time.time() - start_time
        metrics.inference_model_load_duration_seconds.labels(
            model=self.config.id
        ).observe(duration)
        metrics.inference_model_status.labels(model=self.config.id).set(1)

    async def download(self) -> None:
        pass
        # assert self.datadir
        # if self.integrity_marker_file.exists():
        #     log.info(
        #         f"{self.slug}: found valid cached data, no download needed."
        #     )
        #     return
        # log.info(f"{self.slug}: Downloading from hugging face ...")

        # # XXX HF_HUB_DISABLE_PROGRESS_BARS
        # await asyncio.to_thread(
        #     snapshot_download,
        #     repo_id=self.config.repo,
        #     revision=self.config.revision,
        #     cache_dir=self.datadir / ".cache",
        #     local_dir=self.datadir,
        # )
        # self.integrity_marker_file.touch()
        # log.info(f"{self.slug}: success")


class SystemdModel(Model):
    _engine = "systemd"

    def __init__(self, config: SystemdModelConfig):
        super().__init__(config)
        self._config = config

    @property
    def slug(self) -> str:
        assert self.config.id is not None
        return slugify(self.config.id, 64)

    async def start(self) -> None:
        """Start the model process."""
        log.info("Starting model", model=self.config.id)
        assert self.config.id is not None
        assert self.process_status == "stopped"

        start_time = time.time()

        self.process_status = "starting"

        log.info("Starting model via systemd unit", unit=self._config.unit)

        proc = await asyncio.create_subprocess_exec(
            "/run/wrappers/bin/sudo",  # nixos-ism
            "systemctl",
            "start",
            self._config.unit,
        )
        await proc.wait()
        if proc.returncode:
            log.info("Got error exit code", code=proc.returncode)
            await self.terminate()
            return

        log.info("Unit started, waiting for server to respond ...")
        try:
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
        proc = await asyncio.create_subprocess_exec(
            "/run/wrappers/bin/sudo",
            "systemctl",
            "stop",
            self._config.unit,
        )
        await proc.wait()
        self.process_status = "stopped"

    async def download(self) -> None:
        pass
