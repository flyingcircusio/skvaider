import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, AsyncGenerator

import httpx
import structlog
from fastapi import HTTPException

from ..typing import ConfigDict, ConfigValue, JSONObject

if TYPE_CHECKING:
    # Avoid circular imports
    from .models import AIModel
    from .pool import Pool


class ModelConfig:
    """Configuration for model-specific options"""

    # map model names (including or excluding tags) to dicts containing model-specific settings
    config: ConfigDict

    def __init__(self, config: ConfigDict):
        self.config = config

    def get(self, model_id: str) -> ConfigValue:
        """Get custom options for a specific model"""
        for candidate in [model_id, model_id.split(":")[0], "__default__"]:
            if candidate in self.config:
                return self.config[candidate]
        return {}


class Backend(ABC):
    """Connection to a single backend."""

    url: str

    health_interval: int = 15
    healthy: bool = False
    unhealthy_reason: str = ""
    models: dict[str, "AIModel"]

    memory: dict[str, dict[str, int]]

    def __init__(self, url: str):
        self.url = url
        self.models = {}
        self.memory = {}
        self.log = structlog.stdlib.get_logger().bind(backend=self.url)

    @property
    def memory_usage(self):
        # XXX adapt to
        usage = 0
        for model in self.models.values():
            for mem in model.memory_usage.values():
                usage += mem
        return usage

    @abstractmethod
    async def post(self, path: str, data: dict[str, Any]): ...

    @abstractmethod
    def post_stream(
        self, path: str, data: JSONObject
    ) -> AsyncGenerator[str, None]: ...

    @abstractmethod
    async def load_model_with_options(self, model_id: str) -> bool: ...

    @abstractmethod
    async def monitor_health_and_update_models(self, pool: "Pool"): ...


class SkvaiderBackend(Backend):
    async def post(self, path: str, data: dict[str, Any]):
        model_id = data.get("model")
        if not model_id:
            raise HTTPException(status_code=400, detail="Model not specified")

        url = f"{self.url}/models/{model_id}/proxy{path}"

        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.post(url, json=data, timeout=120)
            return r.json()

    async def post_stream(
        self, path: str, data: JSONObject
    ) -> AsyncGenerator[str, None]:
        model_id = data.get("model")
        if not model_id:
            raise HTTPException(status_code=400, detail="Model not specified")

        url = f"{self.url}/models/{model_id}/proxy{path}"

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
                f"{self.url}/models/{model_id}/load",
                timeout=120,
            )
            return r.status_code == 200

    async def monitor_health_and_update_models(self, pool: "Pool") -> None:
        self.log.debug("starting monitor")
        while True:
            try:
                await self._update_usage(pool)
                await self._update_models(pool)

                self.healthy = True
            except Exception as e:
                self.log.error("monitor failed", error=str(e))
                self.healthy = False
                self.unhealthy_reason = str(e)

            await asyncio.sleep(self.health_interval)

    async def _update_models(self, pool: "Pool") -> None:
        from .models import AIModel

        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.get(f"{self.url}/models")
            r.raise_for_status()
            r_json = r.json()
            known_models = r_json["models"]

        self.log.info("updating models")
        current_models = self.models
        updated_models = {}
        for model in known_models:
            if model["id"] not in current_models:
                model_obj = AIModel(
                    id=model["id"],
                    created=0,
                    owned_by="skvaider",
                    backend=self,
                )
            else:
                model_obj = current_models[model["id"]]

            updated_models[model_obj.id] = model_obj

            if "active" in model["status"]:
                model_obj.is_loaded = True
            model_obj.memory_usage = model.get("memory_usage")
            self.log.info(
                "model memory usage",
                model=model_obj.id,
                memory=model_obj.memory_usage,
            )

        self.models = updated_models
        pool.update_model_maps()

    async def _update_usage(self, pool: "Pool") -> None:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.get(f"{self.url}/manager/usage")
            r.raise_for_status()
            usage = r.json()

        self.memory = usage["memory"]

        for backend, m in self.memory.items():
            self.log.info("host memory usage", backend=backend, **m)
