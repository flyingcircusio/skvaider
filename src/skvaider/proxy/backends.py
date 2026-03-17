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
    pool: "Pool"

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
        self.request_health_update = asyncio.Event()

    @abstractmethod
    async def post(self, path: str, data: dict[str, Any]) -> Any: ...

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
    """In-memory backend for tests.

    Simulates model loading with memory accounting and optional
    request failures via ``fail_count``.
    """

    def __init__(self, url: str, fail_count: int = 0):
        super().__init__(url)
        self.fail_count = fail_count
        self.call_count = 0

    async def post(self, path: str, data: dict[str, Any]) -> Any:
        self.call_count += 1
        if self.call_count <= self.fail_count:
            raise HTTPException(status_code=540, detail="Backend unavailable")
        return {"id": "cmpl-1", "choices": []}

    async def post_stream(
        self, path: str, data: dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        self.call_count += 1
        if self.call_count <= self.fail_count:
            raise HTTPException(status_code=540, detail="Backend unavailable")
        yield f"data: chunk from {self.url}\n\n"

    async def load_model(self, model_id: str) -> bool:
        model = self.models[model_id]
        assert not model.is_loaded
        configured = model.configured_memory
        if not set(configured.keys()).issubset(self.memory.keys()):
            return False
        for kind, usage in configured.items():
            if self.memory[kind]["free"] <= usage:
                return False
        for kind, usage in configured.items():
            self.memory[kind]["free"] -= usage
        model.is_loaded = True
        return True

    async def unload_model(self, model_id: str) -> None:
        model = self.models[model_id]
        assert model.is_loaded
        for kind, usage in model.configured_memory.items():
            if kind in self.memory:
                self.memory[kind]["free"] += usage
        model.is_loaded = False

    async def monitor_health_and_update_models(self) -> None:
        # No-op: looping here leaks an asyncio task that outlives the test.
        pass


class SkvaiderBackend(Backend):
    # protect against causing multiple load/unload operations at the same time on this backend
    loading_lock: asyncio.Lock

    def __init__(self, url: str):
        super().__init__(url)
        self.loading_lock = asyncio.Lock()

    async def post(self, path: str, data: dict[str, Any]):
        model_id = data.get("model")
        if not model_id:
            raise HTTPException(status_code=400, detail="Model not specified")

        url = f"{self.url}/models/{model_id}/proxy{path}"

        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.post(url, json=data, timeout=120)
            if r.status_code == 540:
                raise HTTPException(
                    status_code=540, detail="Backend unavailable"
                )
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
                if response.status_code == 540:
                    raise HTTPException(
                        status_code=540, detail="Backend unavailable"
                    )
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
            # XXX make this part of the load/unload protocol to avoid a roundtrip?
            self.request_health_update.set()
            return success

    async def unload_model(self, model_id: str):
        async with self.loading_lock:
            model = self.models[model_id]
            if not model.is_loaded:
                return
            # This makes this model immediately invisible for new requests, so once it's
            # idle we can continue without any further locking.
            model.is_loaded = False
            await model.idle.wait()
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
            # XXX make this part of the load/unload protocol to avoid a roundtrip?
            self.request_health_update.set()

    async def monitor_health_and_update_models(self) -> None:
        self.log.debug("starting monitor")
        while True:
            was_healthy = self.healthy

            in_progress = sum([x.in_progress for x in self.models.values()])
            try:
                if in_progress:
                    # XXX skip health check as we can't communicate out of band at the moment
                    # otherwise we mark busy backends as dead too fast and cause superfluous
                    # unloading of healthy models.
                    pass
                else:
                    async with self.loading_lock:
                        await self._update_usage()
                        await self._update_models()
                        self.healthy = True
            except httpx.ConnectError as e:
                self.log.warning("monitor failed to connect", error=str(e))
                self.healthy = False
                self.unhealthy_reason = str(e)
            except Exception as e:
                self.log.exception("monitor failed", error=repr(e))
                self.healthy = False
                self.unhealthy_reason = repr(e)

            if was_healthy != self.healthy:
                self.log.info(
                    "health state changed",
                    was_healthy=was_healthy,
                    is_healthy=self.healthy,
                )
                self.pool.tasks.create(self.pool.rebalance)

            self.request_health_update.clear()

            self.log.info(
                "waiting for next health check", timeout=self.health_interval
            )
            try:
                await asyncio.wait_for(
                    self.request_health_update.wait(),
                    timeout=self.health_interval,
                )
            except asyncio.TimeoutError:
                pass

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
            model_id = model["id"]
            if model_id not in current_models:
                model_obj = AIModel(
                    id=model_id,
                    created=0,
                    owned_by="skvaider",
                    backend=self,
                )
            else:
                model_obj = current_models[model_id]

            updated_models[model_obj.id] = model_obj

            model_obj.is_loaded = "active" in model["status"]
            model_obj.limit = model["max_requests"]
            model_obj.memory_usage = model.get("memory_usage") or {}

            if model_obj.is_loaded:
                for resource, (
                    actual,
                    configured,
                ) in model_obj.check_memory_usage().items():
                    self.log.warning(
                        "model uses more memory than configured",
                        model=model_id,
                        resource=resource,
                        actual=actual,
                        configured=configured,
                    )

            self.log.info(
                "model status",
                model=model_obj.id,
                actual_memory=model_obj.memory_usage,
                configured_memory=model_obj.configured_memory,
                loaded=model_obj.is_loaded,
            )

        self.models = updated_models
        self.pool.tasks.create(self.pool.rebalance)

    async def _update_usage(self) -> None:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.get(f"{self.url}/manager/usage")
            r.raise_for_status()
            usage = r.json()

        self.memory = usage["memory"]

        for backend, m in self.memory.items():
            self.log.info("host memory usage", backend=backend, **m)
