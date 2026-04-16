import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, AsyncGenerator, Literal

import httpx
import httpx_sse
import structlog
from fastapi import HTTPException
from pydantic import BaseModel

from skvaider.utils import ModelAPI, RequestMethod, RequestModel, ResponseModel

from ..typing import ConfigDict, ConfigValue, JSONObject

if TYPE_CHECKING:
    # Avoid circular imports
    from .models import AIModel
    from .pool import Pool

from skvaider.manifest import ManifestRequest, Serial


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

    current_serial = Serial.floor()

    health_interval: int = 1
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
    async def post(
        self, path: str, data: dict[str, Any], request_id: str = ""
    ) -> Any: ...

    @abstractmethod
    def post_stream(
        self, path: str, data: JSONObject, request_id: str = ""
    ) -> AsyncGenerator[str, None]: ...

    @abstractmethod
    async def update_manifest(self) -> None: ...

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
        self.last_request_id: str = ""

    async def post(
        self, path: str, data: dict[str, Any], request_id: str = ""
    ) -> Any:
        self.call_count += 1
        self.last_request_id = request_id
        if self.call_count <= self.fail_count:
            raise HTTPException(status_code=540, detail="Backend unavailable")
        return {"id": "cmpl-1", "choices": []}

    async def post_stream(
        self, path: str, data: dict[str, Any], request_id: str = ""
    ) -> AsyncGenerator[str, None]:
        self.call_count += 1
        self.last_request_id = request_id
        if self.call_count <= self.fail_count:
            raise HTTPException(status_code=540, detail="Backend unavailable")
        yield f"data: chunk from {self.url}\n\n"

    async def update_manifest(self) -> None:
        if not self.healthy:
            return
        self.current_serial = self.pool.map_serial
        model_ids = self.pool.last_map[self.url]
        for model_id, model in self.models.items():
            if model.is_loaded and model_id not in model_ids:
                for kind, usage in model.configured_memory.items():
                    if kind in self.memory:
                        self.memory[kind]["free"] += usage
                model.is_loaded = False
        for model_id in model_ids:
            if model_id not in self.models:
                continue
            model = self.models[model_id]
            if model.is_loaded:
                continue
            configured = model.configured_memory
            if not set(configured.keys()).issubset(self.memory.keys()):
                continue
            if any(self.memory[k]["free"] < v for k, v in configured.items()):
                continue
            for kind, usage in configured.items():
                self.memory[kind]["free"] -= usage
            model.is_loaded = True

    async def monitor_health_and_update_models(self) -> None:
        # No-op: looping here leaks an asyncio task that outlives the test.
        pass


class SkvaiderBackend(Backend):
    def __init__(self, url: str):
        super().__init__(url)
        self.backend_api = ModelAPI(url)

    async def post(
        self, path: str, data: dict[str, Any], request_id: str = ""
    ) -> Any:
        model_id = data.get("model")
        if not model_id:
            raise HTTPException(status_code=400, detail="Model not specified")

        url = f"{self.url}/models/{model_id}/proxy{path}"
        headers = {"X-Skvaider-Request-ID": request_id} if request_id else {}

        async with httpx.AsyncClient(follow_redirects=True) as client:
            try:
                r = await client.post(
                    url, json=data, headers=headers, timeout=120
                )
            except httpx.TimeoutException as e:
                raise HTTPException(
                    status_code=504, detail="Backend timeout"
                ) from e
            except httpx.ConnectError as e:
                raise HTTPException(
                    status_code=540, detail="Backend unavailable"
                ) from e
            if r.status_code == 540:
                raise HTTPException(
                    status_code=540, detail="Backend unavailable"
                )
            return r.json()

    async def post_stream(
        self, path: str, data: JSONObject, request_id: str = ""
    ) -> AsyncGenerator[str, None]:
        model_id = data.get("model")

        stream_options = data.get("stream_options", {})
        assert isinstance(stream_options, dict)
        stream_options["include_usage"] = True
        data["stream_options"] = stream_options

        if not model_id:
            raise HTTPException(status_code=400, detail="Model not specified")

        url = f"{self.url}/models/{model_id}/proxy{path}"
        headers = {"X-Skvaider-Request-ID": request_id} if request_id else {}

        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                async with client.stream(
                    "POST", url, json=data, headers=headers, timeout=120
                ) as response:
                    if response.status_code == 540:
                        raise HTTPException(
                            status_code=540, detail="Backend unavailable"
                        )
                    if response.status_code >= 400:
                        body = await response.aread()
                        raise HTTPException(
                            status_code=response.status_code,
                            detail=body.decode(errors="replace"),
                        )
                    async for event in httpx_sse.EventSource(
                        response
                    ).aiter_sse():
                        yield f"data: {event.data}\n\n"
        except httpx.TimeoutException as e:
            raise HTTPException(
                status_code=504, detail="Backend timeout"
            ) from e
        except httpx.ConnectError as e:
            raise HTTPException(
                status_code=540, detail="Backend unavailable"
            ) from e

    async def update_manifest(self) -> None:
        # This first part needs to be run atomically, so no async context
        # switches to ensure we have a consistent map and serial.
        if self.current_serial == self.pool.map_serial:
            # The backend is up to date.
            return
        if not self.healthy:
            return
        if self.url not in self.pool.last_map:
            # There is no data for this backend, yet. Leave it be
            # to avoid superfluously unloading models that we might
            # need in a second.
            return
        model_ids = self.pool.last_map[self.url]
        serial = self.pool.map_serial
        self.log.info(
            "backend has stale manifest, reconciling",
            backend_serial=str(self.current_serial),
            pool_serial=str(self.pool.map_serial),
        )
        await self.backend_api(ManifestRequest(models=model_ids, serial=serial))

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
                    await self._update_health()
                    self.healthy = True
            except httpx.ConnectError as e:
                self.healthy = False
                self.unhealthy_reason = str(e)
            except Exception as e:
                self.healthy = False
                self.unhealthy_reason = repr(e)
            else:
                self.unhealthy_reason = ""

            # Handle state changes
            if was_healthy == self.healthy:
                # nothing happened
                pass
            elif self.healthy:
                # we just became healthy
                self.log.info("backend became HEALTHY", backend=self.url)
                await self.pool.tasks.create(self.pool.rebalance)
            elif not self.healthy:
                self.log.info(
                    "backend became UNHEALTHY",
                    backend=self.url,
                    reason=self.unhealthy_reason,
                )
                self.current_serial = Serial.floor()
                await self.pool.tasks.create(self.pool.rebalance)

            # Handle steady states
            if self.healthy:
                await self.ensure_healthy()
            else:
                await self.ensure_unhealthy()

            self.request_health_update.clear()
            try:
                await asyncio.wait_for(
                    self.request_health_update.wait(),
                    timeout=self.health_interval,
                )
            except asyncio.TimeoutError:
                pass

    async def ensure_healthy(self):
        """Perform actions to keep a healthy backend updated and consistent.

        This can be called gratuitously and with relatively high frequency
        in an idempotent/convergent fashion while the backend is in healthy state.

        """
        pass

    async def ensure_unhealthy(self):
        """Perform actions for ensuring consistency in unhealthy state.

        This can be called gratuitously and with relatively high frequency
        in an idempotent/convergent fashion.

        """
        pass

    async def _update_health(self) -> None:
        health = await self.backend_api(BackendHealthRequest())

        assert health.status == "ok"
        self.memory = health.usage
        self.current_serial = health.current_serial

        from .models import AIModel

        current_models = self.models
        updated_models = {}
        for model in health.models:
            model_id = model.id
            if model_id not in self.pool.model_configs:
                # Ignore models that haven't been configured on the proxy.
                continue
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

            model_obj.is_loaded = "active" in model.status
            model_obj.limit = model.max_requests
            model_obj.memory_usage = model.memory_usage or {}

            from .models import CheckResult

            checks: dict[str, CheckResult] = {}

            exceeding = model_obj.check_memory_usage()
            if exceeding:
                over = ", ".join(
                    f"{r}: {actual} > {configured}"
                    for r, (actual, configured) in exceeding.items()
                )
                checks["memory"] = CheckResult(
                    status="warning",
                    message=f"exceeds configured memory: {over}",
                )
            else:
                checks["memory"] = CheckResult(status="ok", message="ok")

            if model_obj.is_loaded:
                for name, message in model.health_checks.items():
                    checks[name] = CheckResult(
                        status="ok" if not message else "critical",
                        message=message or "ok",
                    )
            model_obj.checks = checks
        self.models = updated_models
        # TODO Questionable on high frequency -- maybe only run when memory usage
        # changes beyond a threshold?
        self.pool.tasks.create(self.pool.rebalance)


class BackendModelInfo(BaseModel):
    id: str
    status: set[str]
    max_requests: int
    memory_usage: dict[str, int]
    health_checks: dict[str, str]  # name -> message, "" is OK


class BackendHealthResponse(ResponseModel):
    status: Literal["ok"]

    current_serial: Serial
    usage: dict[str, dict[str, int]]

    models: list[BackendModelInfo]


class BackendHealthRequest(RequestModel[BackendHealthResponse]):
    """Wire format for the manifest PATCH request sent from proxy to inference server."""

    request_method: RequestMethod = "get"
    request_path: str = "/manager/health"
    response_model: type[BackendHealthResponse] = BackendHealthResponse
