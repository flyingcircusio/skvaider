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


class OllamaBackend(Backend):
    async def post(self, path: str, data: dict):
        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.post(self.url + path, json=data, timeout=120)
            return r.json()

    async def post_stream(
        self, path: str, data: dict
    ) -> AsyncGenerator[str, None]:
        """Stream responses from the backend"""
        async with httpx.AsyncClient(follow_redirects=True) as client:
            async with client.stream(
                "POST", self.url + path, json=data, timeout=120
            ) as response:
                async for chunk in response.aiter_text():
                    if chunk.strip():
                        yield chunk

    async def load_model_with_options(self, model_id: str) -> bool:
        """Load a model with custom options if configured"""
        options = self.model_config.get(model_id)
        # Load model with custom options using Ollama's /api/generate endpoint
        load_data = {
            "model": model_id,
            "prompt": "",  # Empty prompt to just load the model
            "options": options,
        }
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                r = await client.post(
                    self.url + "/api/generate", json=load_data, timeout=120
                )
                result = r.json()
                return result.get("done", False)
            await self.update_model_load_status()
        except Exception as e:
            self.log.error("failed loading model", exception=e, model=model_id)
            raise

    async def update_model_load_status(self):
        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.get(self.url + "/api/ps")
            model_status = {}
            for entry in r.json()["models"]:
                model_status[entry["name"]] = entry

        for model_id, model_obj in self.models.items():
            if model_data := model_status.get(model_obj.id):
                model_obj.is_loaded = True
                model_obj.memory_usage = model_data["size_vram"]
            else:
                model_obj.is_loaded = False
                model_obj.memory_usage = 0

    async def monitor_health_and_update_models(self, pool: "Pool"):
        from .models import AIModel

        self.log.debug("starting monitor")
        while True:
            try:
                self.log.debug("probing backend")
                async with httpx.AsyncClient(follow_redirects=True) as client:
                    r = await client.get(self.url + "/v1/models")
                    known_models = r.json()["data"] or ()
                self.log.debug("updating backends")
                current_models = self.models
                updated_models = {}
                for model in known_models:
                    if model["id"] not in current_models:
                        model_obj = AIModel(
                            id=model["id"],
                            created=model["created"],
                            owned_by=model["owned_by"],
                            backend=self,
                        )
                    else:
                        model_obj = current_models.get(
                            model["id"],
                        )
                        model_obj.created = model["created"]
                        model_obj.owned_by = model["owned_by"]

                    updated_models[model_obj.id] = model_obj

                self.models = updated_models

                await self.update_model_load_status()

                pool.update_model_maps()

            except Exception as e:
                if self.healthy or self.healthy is None:
                    self.log.error("marking as unhealthy", error=str(e))
                self.healthy = False
                self.unhealthy_reason = str(e)
                # Reset our model knowledge, drop statistics
                self.models = {}
            else:
                if not self.healthy:
                    self.log.info("marking as healthy")
                self.healthy = True
                self.unhealthy_reason = ""

            await asyncio.sleep(self.health_interval)


class SkvaiderBackend(Backend):
    def __init__(self, url, model_config):
        super().__init__(url, model_config)

    async def post(self, path: str, data: dict):
        model_id = data.get("model")
        if not model_id:
            raise HTTPException(status_code=400, detail="Model not specified")

        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.post(
                f"{self.url}/load", json={"model": model_id}, timeout=120
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
                f"{self.url}/load", json={"model": model_id}, timeout=120
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
                f"{self.url}/load", json={"model": model_id}, timeout=120
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
