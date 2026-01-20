import asyncio
import functools
import itertools
import json
import re
from pathlib import Path
from typing import Callable, Dict, Optional

import anyio
import httpx
import structlog
from pydantic import BaseModel

log = structlog.get_logger()


def locked(func: Callable) -> Callable:
    """Decorator that acquires self._lock before executing an async method."""

    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        async with self._lock:
            return await func(self, *args, **kwargs)

    return wrapper


class ModelConfig(BaseModel):
    name: str
    filename: str
    cmd_args: list[str] = []
    context_size: Optional[int] = None


class RunningModel:
    process: asyncio.subprocess.Process | None = None
    endpoint: str | None = None

    _shutdown = False
    _port_found: asyncio.Event
    _tasks: list[asyncio.Task]
    _host = "127.0.0.1"

    def __init__(self, config: ModelConfig):
        self.config = config
        self._port_found = asyncio.Event()
        self._tasks = []

    async def start(self):
        """Start the model process."""
        log.info("Starting model", model=self.config.name)
        # fmt: off
        cmd = [
            "llama-server",
            "--no-webui",
            "-a", self.config.name,
            "--model", str(self.config.filename),
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
        log.info(
            "Model started", model=self.config.name, endpoint=self.endpoint
        )

    def _create_task(self, awaitable):
        t = asyncio.create_task(awaitable)
        self._tasks.append(t)
        return t

    async def terminate(self):
        """Terminate the process, escalating to kill if necessary."""
        log.info("Terminating model process", model=self.config.name)
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
                    model=self.config.name,
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
            model=self.config.name,
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
                model=self.config.name,
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


class ModelManager:
    def __init__(self, models_dir: Path = Path("models")):
        self.models_dir = models_dir
        self.running_models: Dict[str, RunningModel] = {}
        self._lock = asyncio.Lock()

    async def list_models(self) -> list[str]:
        models = []
        if not self.models_dir.exists():
            return models

        # Allow max one layer of hierarchy
        files = itertools.chain(
            self.models_dir.glob("*.json"),
            self.models_dir.glob("*/*.json"),
        )

        for meta_file in files:
            try:
                async with await anyio.open_file(meta_file, "r") as f:
                    content = await f.read()
                data = json.loads(content)
                if name := data.get("name"):
                    models.append(name)
            except Exception:
                pass
        return models

    async def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        # Scan models directory for matching metadata
        # This is inefficient for many models, but fine for now.
        # We assume metadata files are named <filename>.json
        for meta_file in self.models_dir.glob("*/*.json"):
            try:
                async with await anyio.open_file(meta_file, "r") as f:
                    content = await f.read()
                data = json.loads(content)
                # Check if this metadata corresponds to the requested model name
                # We assume the metadata has a "name" field which is the public ID
                if data.get("name") == model_name:
                    return ModelConfig(
                        name=model_name,
                        filename=data["filename"],
                        cmd_args=data.get("cmd_args", []),
                        context_size=data["context_size"],
                    )
            except Exception as e:
                log.warn(
                    "Failed to read metadata file",
                    file=str(meta_file),
                    error=str(e),
                )
                raise
        raise KeyError(model_name)

    @locked
    async def get_or_start_model(
        self, model_name: str, timeout: int = 60
    ) -> Optional[RunningModel]:
        if model_name in self.running_models:
            model = self.running_models[model_name]
            if not model._shutdown:
                return model
            else:
                self.running_models.pop(model_name)

        config = await self.get_model_config(model_name)
        model = RunningModel(config)
        try:
            await asyncio.wait_for(model.start(), timeout=timeout)
        except (
            asyncio.TimeoutError
        ):  # XXX the timeout might need to be model specific?
            log.error("Timeout starting model", model=model_name)
            await model.terminate()
            raise
        self.running_models[model_name] = model
        return model

    @locked
    async def unload_model(self, model_name: str):
        if model_name in self.running_models:
            model = self.running_models[model_name]
            await model.terminate()
            del self.running_models[model_name]

    @locked
    async def shutdown(self):
        for model in self.running_models.values():
            await model.terminate()
        self.running_models.clear()
