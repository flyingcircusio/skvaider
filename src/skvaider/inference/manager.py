import asyncio
import functools
import hashlib
import itertools
import json
import re
from pathlib import Path
from typing import Callable, Dict, Optional

import anyio
import httpx
import structlog

import skvaider.inference.config
from skvaider.utils import slugify

log = structlog.get_logger()


def locked(func: Callable) -> Callable:
    """Decorator that acquires self._lock before executing an async method."""

    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        async with self._lock:
            return await func(self, *args, **kwargs)

    return wrapper


class Model:
    process: asyncio.subprocess.Process | None = None
    endpoint: str | None = None
    config: "skvaider.inference.config.Model"
    datadir: Path

    _shutdown = False
    _port_found: asyncio.Event
    _tasks: list[asyncio.Task]
    _host = "127.0.0.1"

    def __init__(self, config):
        self.config = config

        self._port_found = asyncio.Event()
        self._tasks = []

    @property
    def slug(self):
        slug = slugify(self.config.id, 64)
        # The hash as suffix to assist auto-completion in shells
        slug += "-" + self.config.hash[:8]
        return slug

    @property
    def model_file(self):
        return self.datadir / "model.gguf"

    @property
    def integrity_marker_file(self):
        return self.datadir / "integrity.ok"

    async def download(self):
        assert self.datadir
        if self.integrity_marker_file.exists():
            log.info(
                f"{self.slug}: found valid cached data, no download needed."
            )
            return

        if self.model_file.exists():
            self.model_file.unlink()

        log.info(f"Downloading {self.config.url} ...")
        got_hash_ = hashlib.sha256()
        async with httpx.AsyncClient(timeout=30) as client:
            async with client.stream(
                "GET", self.config.url, follow_redirects=True
            ) as response:
                download_size = int(response.headers["content-length"])
                download_status = 0
                response.raise_for_status()

                async def log_progress():
                    while True:
                        progress = int((download_status / download_size) * 100)
                        log.info(f"{self.slug}: {progress}%")
                        await asyncio.sleep(1)

                status_task = asyncio.create_task(log_progress())
                try:
                    async with await anyio.open_file(
                        self.model_file, "wb"
                    ) as f:
                        async for chunk in response.aiter_bytes():
                            download_status += len(chunk)
                            got_hash_.update(chunk)
                            await f.write(chunk)
                finally:
                    status_task.cancel()

        got_hash = got_hash_.hexdigest()

        if self.config.hash != got_hash:
            log.error(
                f"Hash error downloading {self.config.id}: expected {self.config.hash}, got {got_hash}."
            )
            raise ValueError(got_hash)

        self.integrity_marker_file.touch()
        log.info(f"{self.slug}: success")

    async def start(self):
        """Start the model process."""
        log.info("Starting model", model=self.config.id)
        # fmt: off
        cmd = [
                "llama-server",
                "--no-webui",
                "-a", self.config.id,
                "--model", str(self.model_file),
                "--jinja", # XXX per model?
                "--host", self._host,
                "--port", "0",  # let the kernel select a free port
                # Monitoring
                "--metrics",
                "--slots",
            ]
        if self.config.context_size:
            cmd += [
                    "--ctx-size", str(self.config.context_size),
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
        except Exception:
            await self.terminate()
            raise
        log.info("Model started", model=self.config.id, endpoint=self.endpoint)

    def _create_task(self, awaitable):
        t = asyncio.create_task(awaitable)
        self._tasks.append(t)
        return t

    async def terminate(self):
        """Terminate the process, escalating to kill if necessary."""
        log.info("Terminating model process", model=self.config.id)
        self._shutdown = True
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
                return

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

    async def _monitor_process(self):
        """Monitor whether our process has exited."""
        assert self.process is not None
        await self.process.wait()
        if self._shutdown:
            return
        log.error(
            "Process exited unexpectedly",
            model=self.config.id,
            returncode=self.process.returncode,
        )
        # Clean up
        await self.terminate()

    async def _monitor_output(self, stream, is_stderr: bool):
        stream_name = "stderr" if is_stderr else "stdout"
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
                    r"main: HTTP server is listening, hostname: .*, port: (\d+)",
                    line_str,
                )
                if match:
                    port = int(match.group(1))
                    self.endpoint = f"http://{self._host}:{port}"
                    self._port_found.set()

    async def _wait_for_startup(self):
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
    models: Dict[str, Model]

    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.models = {}
        self._lock = asyncio.Lock()

    def add_model(self, model):
        assert model.config.id not in self.models
        self.models[model.config.id] = model
        model.datadir = self.models_dir / model.slug
        model.datadir.mkdir(exist_ok=True)

    async def list_models(self) -> list[str]:
        return list(self.models.keys())

    @locked
    async def get_or_start_model(
        self,
        model_name: str,
        timeout: int = 60,
    ) -> Optional[Model]:
        model = self.models[model_name]
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
    async def unload_model(self, model_name: str):
        model = self.models[model_name]
        await model.terminate()

    @locked
    async def shutdown(self):
        for model in self.models.values():
            await model.terminate()
        self.models.clear()
