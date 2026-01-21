import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Optional

import httpx
import structlog
from fastapi import HTTPException

if TYPE_CHECKING:
    from .models import AIModel

if TYPE_CHECKING:
    from .pool import Pool


class ModelConfig:
    """Configuration for model-specific options"""

    # map model names (including or excluding tags) to dicts containing model-specific settings
    config: Dict[str, Dict[str, Any]]

    def __init__(self, config):
        self.config = config

    def get(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get custom options for a specific model"""
        for candidate in [model_id, model_id.split(":")[0], "__default__"]:
            if candidate in self.config:
                return self.config[candidate]
        return {}


class Backend(ABC):
    """Connection to a single backend."""

    url: str

    health_interval: int = 15
    healthy: bool = None
    unhealthy_reason: str = ""
    models: dict[str, "AIModel"]
    model_config: ModelConfig

    def __init__(self, url, model_config):
        self.url = url
        self.models = {}
        self.model_config = model_config
        self.log = structlog.stdlib.get_logger().bind(backend=self.url)

    @property
    def memory_usage(self):
        return sum([v.memory_usage for v in self.models.values()])

    @abstractmethod
    async def post(self, path: str, data: dict): ...

    @abstractmethod
    async def post_stream(
        self, path: str, data: dict
    ) -> AsyncGenerator[str, None]: ...

    @abstractmethod
    async def load_model_with_options(self, model_id: str) -> bool: ...

    @abstractmethod
    async def monitor_health_and_update_models(self, pool: "Pool"): ...


class SkvaiderBackend(Backend):
    def __init__(self, url, model_config):
        super().__init__(url, model_config)

    async def post(self, path: str, data: dict):
        model_id = data.get("model")
        if not model_id:
            raise HTTPException(status_code=400, detail="Model not specified")

        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.post(
                f"{self.url}/get_running_model_or_load",
                json={"model": model_id},
                timeout=120,
            )
            if r.status_code == 404:
                raise HTTPException(
                    status_code=404, detail=f"Model {model_id} not found"
                )
            if r.status_code != 200:
                raise HTTPException(
                    status_code=500, detail=f"Failed to load model: {r.text}"
                )
            port = r.json()["port"]

        url = f"http://localhost:{port}{path}"

        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.post(url, json=data, timeout=120)
            return r.json()

    async def post_stream(
        self, path: str, data: dict
    ) -> AsyncGenerator[str, None]:
        model_id = data.get("model")
        if not model_id:
            raise HTTPException(status_code=400, detail="Model not specified")

        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.post(
                f"{self.url}/get_running_model_or_load",
                json={"model": model_id},
                timeout=120,
            )
            if r.status_code == 404:
                raise HTTPException(
                    status_code=404, detail=f"Model {model_id} not found"
                )
            if r.status_code != 200:
                raise HTTPException(
                    status_code=500, detail=f"Failed to load model: {r.text}"
                )
            port = r.json()["port"]

        url = f"http://localhost:{port}{path}"

        async with httpx.AsyncClient(follow_redirects=True) as client:
            async with client.stream(
                "POST", url, json=data, timeout=120
            ) as response:
                async for chunk in response.aiter_text():
                    if chunk.strip():
                        yield chunk

    async def load_model_with_options(self, model_id: str) -> bool:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.post(
                f"{self.url}/get_running_model_or_load",
                json={"model": model_id},
                timeout=120,
            )
            return r.status_code == 200

    async def monitor_health_and_update_models(self, pool: "Pool"):
        from .models import AIModel

        self.log.debug("starting monitor")
        while True:
            try:
                async with httpx.AsyncClient(follow_redirects=True) as client:
                    r = await client.get(f"{self.url}/models")
                    known_models = r.json()["models"]

                    r_running = await client.get(f"{self.url}/running_models")
                    running_models = r_running.json()["models"]

                self.log.debug("updating backends")
                current_models = self.models
                updated_models = {}
                for model_name in known_models:
                    if model_name not in current_models:
                        model_obj = AIModel(
                            id=model_name,
                            created=0,
                            owned_by="skvaider",
                            backend=self,
                        )
                    else:
                        model_obj = current_models.get(model_name)

                    updated_models[model_obj.id] = model_obj

                    if model_name in running_models:
                        model_obj.is_loaded = True
                        model_obj.memory_usage = 0
                    else:
                        model_obj.is_loaded = False
                        model_obj.memory_usage = 0

                self.models = updated_models
                pool.update_model_maps()
                self.healthy = True

            except Exception as e:
                self.log.error("monitor failed", error=str(e))
                self.healthy = False
                self.unhealthy_reason = str(e)

            await asyncio.sleep(self.health_interval)
