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

    def __init__(self, url: str):
        self.url = url
        self.models = {}
        self.log = structlog.stdlib.get_logger().bind(backend=self.url)

    @property
    def memory_usage(self):
        return sum([v.memory_usage for v in self.models.values()])

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

    async def monitor_health_and_update_models(self, pool: "Pool"):
        from .models import AIModel

        self.log.debug("starting monitor")
        while True:
            try:
                async with httpx.AsyncClient(follow_redirects=True) as client:
                    r = await client.get(f"{self.url}/models")
                    r_json = r.json()
                    known_models = r_json["models"]

                self.log.debug("updating backends")
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
