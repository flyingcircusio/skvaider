import asyncio
import functools
import hashlib
import re
import shutil
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, Concatenate, Literal, ParamSpec, Protocol, TypeVar

import anyio
import httpx
import structlog

import skvaider.inference.config
from skvaider.utils import slugify

log = structlog.get_logger()

P = ParamSpec("P")
R = TypeVar("R")


class HasLock(Protocol):
    _lock: asyncio.Lock


SelfT = TypeVar("SelfT", bound=HasLock)


def locked(
    func: Callable[Concatenate[SelfT, P], Coroutine[Any, Any, R]],
) -> Callable[Concatenate[SelfT, P], Coroutine[Any, Any, R]]:
    """Decorator that acquires self._lock before executing an async method."""

    @functools.wraps(func)
    async def wrapper(self: SelfT, *args: P.args, **kwargs: P.kwargs) -> R:
        async with self._lock:  # pyright: ignore[reportPrivateUsage]
            return await func(self, *args, **kwargs)

    return wrapper


class Model:
    process: asyncio.subprocess.Process | None = None
    endpoint: str | None = None
    config: "skvaider.inference.config.ModelConfig"
    datadir: Path
    process_status: Literal["stopped", "running", "starting", "stopping"] = (
        "stopped"
    )
    health_status: Literal["healthy", "unhealthy", ""] = ""
    health_check_interval: float = 30
    health_check_timeout: float = 10
    verification_data: dict[str, list[float]] | None = None

    _port_found: asyncio.Event
    _tasks: list[asyncio.Task[Any]]
    _host = "127.0.0.1"

    async def _check_health(self) -> bool: ...

    def __init__(self, config: "skvaider.inference.config.ModelConfig"):
        self.config = config

        self._port_found = asyncio.Event()
        self._tasks = []

        if self.is_embedding:
            self._check_health = self._check_embedding_health
        else:
            self._check_health = self._check_completion_health

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

                    status_task = asyncio.create_task(log_progress())
                    try:
                        async with await anyio.open_file(model_file, "ab") as f:
                            async for chunk in response.aiter_bytes():
                                download_status += len(chunk)
                                got_hash_.update(chunk)
                                await f.write(chunk)
                    finally:
                        status_task.cancel()

            got_hash = got_hash_.hexdigest()
            verify_got_hashes.append(got_hash)

        for expected, got in zip(
            [file.hash for file in self.config.files], verify_got_hashes
        ):
            if expected != got:
                raise ValueError(got)

        self.integrity_marker_file.touch()
        log.info(f"{self.slug}: success")

    async def start(self) -> None:
        """Start the model process."""
        assert self.config.id is not None
        log.info("Starting model", model=self.config.id)
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
            self._monitor_process_task = asyncio.create_task(
                self._monitor_process()
            )
            self._create_task(self._monitor_output(self.process.stderr, True))
            self._create_task(self._monitor_output(self.process.stdout, False))
            startup_task = self._create_task(self._wait_for_startup())
            await startup_task
            self._create_task(self._monitor_health())

            # health_status is set by the monitoring task which starts immediately
            # XXX wait here for us to see the active flag? might not be required to block on, though ...
        except Exception:
            await self.terminate()
            raise
        log.info("Model started", model=self.config.id, endpoint=self.endpoint)
        self.process_status = "running"

    def _create_task(
        self, awaitable: Coroutine[Any, Any, Any]
    ) -> asyncio.Task[Any]:
        t = asyncio.create_task(awaitable)
        self._tasks.append(t)
        return t

    async def terminate(self) -> None:
        """Terminate the process, escalating to kill if necessary."""
        log.info("Terminating model process", model=self.config.id)
        self.process_status = "stopping"

        # Cancel the monitor task (not in self._tasks)
        if hasattr(self, "_monitor_process_task"):
            self._monitor_process_task.cancel()
            try:
                await self._monitor_process_task
            except asyncio.CancelledError:
                pass

        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()

        if self.process:
            try:
                self.process.terminate()
            except ProcessLookupError:
                pass
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
        self._port_found.clear()
        self.process = None
        self.endpoint = None
        self.process_status = "stopped"
        self.health_status = ""

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
        while True:
            # We check first, then sleep
            if "running" not in self.status or not self.endpoint:
                self.health_status = ""
                await asyncio.sleep(0.1)
                continue

            try:
                self.health_status = (
                    "healthy" if (await self._check_health()) else "unhealthy"
                )
            except Exception as e:
                log.warning(
                    "Health check error", model=self.config.id, error=str(e)
                )
                self.health_status = "unhealthy"

            await asyncio.sleep(self.health_check_interval)

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

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.models = {}
        self._lock = asyncio.Lock()

    def add_model(self, model: Model) -> None:
        assert model.config.id is not None
        assert model.config.id not in self.models
        self.models[model.config.id] = model
        model.datadir = self.models_dir / model.slug
        model.datadir.mkdir(exist_ok=True)

    def list_models(self) -> list[Model]:
        return list(self.models.values())

    @locked
    async def get_or_start_model(
        self,
        model_name: str,
        timeout: int = 60,
    ) -> Model:
        model = self.models[model_name]
        if "active" not in model.status:
            try:
                await asyncio.wait_for(model.start(), timeout=timeout)
            except (
                asyncio.TimeoutError
            ):  # XXX the timeout might need to be model specific?
                log.error("Timeout starting model", model=model_name)
                await model.terminate()
                raise
        return model

    @locked
    async def unload_model(self, model_name: str) -> None:
        model = self.models[model_name]
        if model.process_status in ["running", "starting"]:  # idempotent
            await model.terminate()

    @locked
    async def shutdown(self) -> None:
        for model in self.models.values():
            await model.terminate()
        self.models.clear()
