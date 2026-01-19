import asyncio
import functools
import itertools
import json
import re
import time
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
    port: int | None = None

    def __init__(self, config: ModelConfig, models_dir: Path):
        self.config = config
        self.models_dir = models_dir

    async def start(self):
        """Start the model process."""
        log.info("Starting model", model=self.config.name)
        try:
            # fmt: off
            cmd = [
                "llama-server",
                "--model", str(self.models_dir / self.config.filename),
                "--port", str(0),  # 0 means to select a random free port
            ]
            if self.config.context_size:
                cmd += ["--ctx-size", str(config.context_size)]
            cmd += self.config.cmd_args
            # fmt: on
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            port_future = asyncio.get_running_loop().create_future()
            asyncio.create_task(self._monitor_output(port_future))

            self.port = await asyncio.wait_for(port_future, timeout=30)
            log.info("Model started", model=self.config.name, port=self.port)

            await self._wait_for_startup()
        except Exception as e:
            log.error(
                "Failed to start model", model=self.config.name, error=str(e)
            )
            if self.process:
                await self._terminate_process()
            raise e

    async def _terminate_process(self):
        """Terminate the process, escalating to kill if necessary."""
        log.info("Terminating model process", model=self.config.name)
        try:
            self.process.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(self.process.wait(), timeout=5)
        except asyncio.TimeoutError:
            log.info(
                "Killing unresponsive model process", model=self.config.name
            )
            try:
                self.process.kill()
            except ProcessLookupError:
                pass

    async def _monitor_output(self, port_future: asyncio.Future):
        async def read_stream(stream, is_stderr):
            while True:
                line = await stream.readline()
                if not line:
                    break
                line_str = line.decode("utf-8", errors="replace").strip()
                if line_str:
                    log.debug(
                        "llama-server",
                        model=self.config.name,
                        stream="stderr" if is_stderr else "stdout",
                        line=line_str,
                    )
                    if is_stderr and not port_future.done():
                        match = re.search(
                            r"main: HTTP server is listening, hostname: .*, port: (\d+)",
                            line_str,
                        )
                        if match:
                            port_future.set_result(int(match.group(1)))

        await asyncio.gather(
            read_stream(self.process.stderr, True),
            read_stream(self.process.stdout, False),
        )

        if not port_future.done():
            port_future.set_exception(
                RuntimeError("Process exited before port was found")
            )

    async def _wait_for_startup(self, timeout: int = 30):
        start = time.time()
        async with httpx.AsyncClient() as client:
            while time.time() - start < timeout:
                try:
                    resp = await client.get(
                        f"http://localhost:{self.port}/health"
                    )
                    if resp.status_code == 200:
                        return
                except Exception:
                    pass
                await asyncio.sleep(0.5)
        raise RuntimeError(f"Model failed to start on port {self.port}")

    async def terminate(self):
        """Terminate this model's process."""
        log.info("Unloading model", model=self.config.name)
        try:
            self.process.terminate()
            await self.process.wait()
        except Exception as e:
            # XXX well ... this likely needs more recovery code. Is the process
            # still running? are we hogging/leaking GPU memory now?
            log.error(
                "Error stopping model", model=self.config.name, error=str(e)
            )


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
        if not self.models_dir.exists():
            return None

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
                # Check if this metadata corresponds to the requested model name
                # We assume the metadata has a "name" field which is the public ID
                if data.get("name") == model_name:
                    return ModelConfig(
                        name=model_name,
                        filename=data.get("filename", meta_file.stem),
                        cmd_args=data.get("cmd_args", []),
                        context_size=data.get("context_size", None),
                    )
            except Exception as e:
                log.warn(
                    "Failed to read metadata file",
                    file=str(meta_file),
                    error=str(e),
                )
                continue
        return None

    @locked
    async def get_or_start_model(
        self, model_name: str
    ) -> Optional[RunningModel]:
        if model_name in self.running_models:
            return self.running_models[model_name]

        config = await self.get_model_config(model_name)
        if not config:
            return None

        model = RunningModel(config, self.models_dir)
        await model.start()
        self.running_models[model_name] = model
        return model

    @locked
    async def unload_model(self, model_name: str):
        if model_name in self.running_models:
            model = self.running_models[model_name]
            await model.terminate()
            del self.running_models[model_name]
