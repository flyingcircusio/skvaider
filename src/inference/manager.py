import asyncio
import json
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
        self.port_range = range(8001, 9000)
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

            port = self._find_free_port()
            if not port:
                raise RuntimeError("No free ports available")

            log.info("Starting model", model=model_name, port=port)

            # Construct command
            # Assuming llama-server is in PATH or we need a config for it
            cmd = [
                "llama-server",
                "-m",
                str(self.models_dir / config.filename),
                "--port",
                str(port),
                "-c",
                str(config.context_size),
            ]
            cmd.extend(config.cmd_args)

            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                # Wait a bit for it to start? Or rely on health check?
                # Let's assume it takes a moment.
                # A better way is to poll the health endpoint of the new process.
                await self._wait_for_startup(port)

                running_model = RunningModel(config, process, port)
                self.running_models[model_name] = running_model
                return running_model
            except Exception as e:
                log.error(
                    "Failed to start model", model=model_name, error=str(e)
                )
                if "process" in locals():
                    try:
                        process.kill()
                    except Exception:
                        pass
                raise

    def _find_free_port(self) -> Optional[int]:
        used_ports = {m.port for m in self.running_models.values()}
        for port in self.port_range:
            if port not in used_ports:
                # TODO: Check if port is actually free on OS level
                return port
        return None

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
