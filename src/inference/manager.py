import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

import anyio
import httpx
import structlog
from pydantic import BaseModel

log = structlog.get_logger()


class ModelConfig(BaseModel):
    name: str
    filename: str
    cmd_args: list[str] = []
    context_size: int = 2048
    idle_timeout: int = 300  # 5 minutes


class RunningModel:
    def __init__(
        self,
        config: ModelConfig,
        process: asyncio.subprocess.Process,
        port: int,
    ):
        self.config = config
        self.process = process
        self.port = port
        self.last_access = time.time()

    def touch(self):
        self.last_access = time.time()


class ModelManager:
    def __init__(self, models_dir: Path = Path("models")):
        self.models_dir = models_dir
        self.running_models: Dict[str, RunningModel] = {}
        self._lock = asyncio.Lock()

    async def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        # Scan models directory for matching metadata
        # This is inefficient for many models, but fine for now.
        # We assume metadata files are named <filename>.json
        if not self.models_dir.exists():
            return None

        for meta_file in self.models_dir.glob("*.json"):
            try:
                async with await anyio.open_file(meta_file, "r") as f:
                    content = await f.read()
                data = json.loads(content)
                # Check if this metadata corresponds to the requested model name
                # We assume the metadata has a "name" field which is the public ID
                if data.get("name") == model_name:
                    # Found it. Now we need the filename.
                    # The metadata file is <filename>.json, so filename is stem
                    filename = meta_file.stem
                    return ModelConfig(
                        name=model_name,
                        filename=filename,
                        cmd_args=data.get("cmd_args", []),
                        context_size=data.get("context_size", 2048),
                        idle_timeout=data.get("idle_timeout", 300),
                    )
            except Exception as e:
                log.warn(
                    "Failed to read metadata file",
                    file=str(meta_file),
                    error=str(e),
                )
                continue
        return None

    async def get_or_start_model(
        self, model_name: str
    ) -> Optional[RunningModel]:
        async with self._lock:
            if model_name in self.running_models:
                model = self.running_models[model_name]
                model.touch()
                return model

            config = await self.get_model_config(model_name)
            if not config:
                return None

            log.info("Starting model", model=model_name)

            # Construct command
            # Assuming llama-server is in PATH or we need a config for it
            cmd = [
                "llama-server",
                "-m",
                str(self.models_dir / config.filename),
                "--port",
                "0",
                "-c",
                str(config.context_size),
                *config.cmd_args,
            ]

            process = None
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                port_future = asyncio.get_running_loop().create_future()
                asyncio.create_task(
                    self._monitor_output(process, model_name, port_future)
                )

                port = await asyncio.wait_for(port_future, timeout=30)
                log.info("Model started", model=model_name, port=port)

                await self._wait_for_startup(port)

                running_model = RunningModel(config, process, port)
                self.running_models[model_name] = running_model
                return running_model
            except Exception as e:
                log.error(
                    "Failed to start model", model=model_name, error=str(e)
                )
                if process:
                    log.info(
                        "Terminating failed model process", model=model_name
                    )
                    try:
                        process.terminate()
                    except ProcessLookupError:
                        pass
                    # wait with timeout
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        log.info(
                            "Killing unresponsive model process",
                            model=model_name,
                        )
                        try:
                            process.kill()
                        except ProcessLookupError:
                            pass
                raise e

    async def _monitor_output(
        self,
        process: asyncio.subprocess.Process,
        model_name: str,
        port_future: asyncio.Future,
    ):
        async def read_stream(stream, is_stderr):
            while True:
                line = await stream.readline()
                if not line:
                    break
                line_str = line.decode("utf-8", errors="replace").strip()
                if line_str:
                    # log.debug("llama-server", model=model_name, stream="stderr" if is_stderr else "stdout", line=line_str)
                    if is_stderr and not port_future.done():
                        match = re.search(
                            r"HTTP server is listening, hostname: .*, port: (\d+)",
                            line_str,
                        )
                        if match:
                            port_future.set_result(int(match.group(1)))

        await asyncio.gather(
            read_stream(process.stderr, True),
            read_stream(process.stdout, False),
        )

        if not port_future.done():
            port_future.set_exception(
                RuntimeError("Process exited before port was found")
            )

    async def _wait_for_startup(self, port: int, timeout: int = 30):
        start = time.time()
        async with httpx.AsyncClient() as client:
            while time.time() - start < timeout:
                try:
                    resp = await client.get(f"http://localhost:{port}/health")
                    if resp.status_code == 200:
                        return
                except Exception:
                    pass
                await asyncio.sleep(0.5)
        raise RuntimeError(f"Model failed to start on port {port}")

    async def check_idle_models(self):
        while True:
            await asyncio.sleep(10)
            now = time.time()
            to_remove = []
            async with self._lock:
                for name, model in self.running_models.items():
                    if now - model.last_access > model.config.idle_timeout:
                        log.info("Unloading idle model", model=name)
                        try:
                            model.process.terminate()
                            await model.process.wait()
                        except Exception as e:
                            log.error(
                                "Error stopping model", model=name, error=str(e)
                            )
                        to_remove.append(name)

                for name in to_remove:
                    del self.running_models[name]
