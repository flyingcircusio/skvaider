import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, AsyncGenerator

import httpx
import structlog
from fastapi import HTTPException

from skvaider import utils

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

    def __init__(self, url: str, pool: "Pool"):
        self.url = url
        self.pool = pool
        self.models = {}
        self.memory = {}
        self.log = structlog.stdlib.get_logger().bind(backend=self.url)

    @abstractmethod
    async def post(self, path: str, data: dict[str, Any]): ...

    @abstractmethod
    def post_stream(
        self, path: str, data: JSONObject
    ) -> AsyncGenerator[str, None]: ...

    @abstractmethod
    async def load_model(self, model_id: str) -> bool: ...

    @abstractmethod
    async def unload_model(self, model_id: str): ...

    @abstractmethod
    async def monitor_health_and_update_models(self): ...


class DummyBackend(Backend):
    async def post(self, path: str, data: dict[str, Any]):
        pass

    async def post_stream(
        self, path: str, data: JSONObject
    ) -> AsyncGenerator[str, None]:
        yield ""

    async def load_model(self, model_id: str) -> bool:
        model = self.models[model_id]
        assert not model.is_loaded
        if not set(model.memory_usage.keys()).issubset(self.memory.keys()):
            return False
        for kind, usage in model.memory_usage.items():
            if self.memory[kind]["free"] <= usage:
                return False
        for kind, usage in model.memory_usage.items():
            self.memory[kind]["free"] -= usage
        model.is_loaded = True
        model.last_used = utils.now()
        return True

    async def unload_model(self, model_id: str):
        model = self.models[model_id]
        assert model.is_loaded
        for kind, usage in model.memory_usage.items():
            if kind in self.memory:
                self.memory[kind]["free"] += usage
        model.is_loaded = False
        model.last_used = utils.datetime_min

    async def monitor_health_and_update_models(self):
        while True:
            await asyncio.sleep(5)


class SkvaiderBackend(Backend):
    # protect against causing multiple load/unload operations at the same time on this backend
    loading_lock: asyncio.Lock

    def __init__(self, url: str, pool: "Pool"):
        super().__init__(url, pool)
        self.loading_lock = asyncio.Lock()

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

    async def load_model(self, model_id: str) -> bool:
        """Load a model on this backend.

        Return True on success and False on failure.

        This double checks whether the backend is considered having sufficient
        free space as this may have changed while waiting for the lock.

        """
        async with self.loading_lock:
            # Only try loading one model at a time on a backend.
            if self.models[model_id].is_loaded:
                return True
            if self.models[model_id].fit_score() == 0:
                # Someone may have loaded a different model in between, so if the score
                # dropped to 0 we need to abort.
                return False
            self.log.info(
                "loading model",
                model=model_id,
                backend=self.url,
                score=self.models[
                    model_id
                ].fit_score(),  # this is a bit weird to log this later, the score might have changed. we really might need to pick up the lock outside, or refactor
            )
            # XXX double check whether this model still fits, as other models might have loaded -
            # alternatively the lock needs to be held elsewhere, too.
            success = False
            async with httpx.AsyncClient(follow_redirects=True) as client:
                try:
                    r = await client.post(
                        f"{self.url}/models/{model_id}/load",
                        timeout=120,
                    )
                except httpx.HTTPError as exc:
                    self.log.error(
                        f"Loading model failed: HTTP Exception for {exc.request.url} - {exc}"
                    )
                else:
                    success = r.status_code == 200
            # XXX make this part of the load/ protocol to avoid a roundtrip?
            self.models[model_id].last_used = utils.now()
            await self._update_usage()
            await self._update_models()
            return success

    async def unload_model(self, model_id: str):
        async with self.loading_lock:
            model = self.models[model_id]
            if not model.is_loaded:
                return
            await model.wait()
            # Immediately mark as non-idle but also unloaded to avoid it being used for further requests.
            # XXX I'm seeing the potential for race conditions here. We might need to zoom out to review
            # the overall logic and coordination within the manager async tasks, the requests being processed,
            # and the way the inference server handles them.
            # Potential issues:
            # - requests being assigned to an unloaded backend/model
            # - getting in some weird kind of deadlock
            #
            # Overall I'd like to keep as few locks as possible and avoid having locks on the inference side, completely.
            model.is_loaded = False
            model.idle = False
            self.log.info("unloading model", model=model_id, backend=self.url)
            async with httpx.AsyncClient(follow_redirects=True) as client:
                try:
                    r = await client.post(
                        f"{self.url}/models/{model_id}/unload",
                        timeout=120,
                    )
                    r.raise_for_status()
                except httpx.HTTPError as exc:
                    self.log.error(
                        f"Unloading model failed: HTTP Exception for {exc.request.url} - {exc}"
                    )
            # XXX make this part of the load/ protocol to avoid a roundtrip?
            model.last_used = utils.datetime_min
            model.idle = True
            await self._update_usage()
            await self._update_models()

    async def monitor_health_and_update_models(self) -> None:
        self.log.debug("starting monitor")
        while True:
            try:
                async with self.loading_lock:
                    await self._update_usage()
                    await self._update_models()
                    self.healthy = True
            except Exception as e:
                self.log.error("monitor failed", error=str(e))
                self.healthy = False
                self.unhealthy_reason = str(e)

            await asyncio.sleep(self.health_interval)

    async def _update_models(self) -> None:
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

            model_obj.is_loaded = "active" in model["status"]
            model_obj.memory_usage = model.get("memory_usage")
            self.log.info(
                "model memory usage",
                model=model_obj.id,
                memory=model_obj.memory_usage,
                loaded=model_obj.is_loaded,
            )

        self.models = updated_models
        self.pool.update_model_maps()

    async def _update_usage(self) -> None:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.get(f"{self.url}/manager/usage")
            r.raise_for_status()
            usage = r.json()

        self.memory = usage["memory"]

        for backend, m in self.memory.items():
            self.log.info("host memory usage", backend=backend, **m)
