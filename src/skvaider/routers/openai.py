"""Open-AI compatible API based on Ollama.

This uses Ollama-internal APIs for better load-balancing but exposes a pure OpenAI-compatible API.

"""

import asyncio
import contextlib
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, Generic, Optional, TypeVar

import httpx
import structlog
import svcs
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from skvaider import utils

router = APIRouter()

T = TypeVar("T")

log = structlog.stdlib.get_logger()


class AIModel(BaseModel):
    """Model object per backend."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    object: str = "model"
    created: int = 0
    owned_by: str

    backend: "Backend" = Field(exclude=True)
    last_used: float = Field(default=0, exclude=True)
    in_progress: int = Field(default=0, exclude=True)
    limit: int = Field(default=5, exclude=True)
    idle: asyncio.Event = Field(default=True, exclude=True)
    is_loaded: bool = Field(default=False, exclude=True)
    memory_usage: int = Field(default=0, exclude=True)
    log: Any = Field(default=None, exclude=True)

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.idle = asyncio.Event()
        self.idle.set()

        self.log = log.bind(model=self.id, backend=self.backend.url)

    @contextlib.asynccontextmanager
    async def use(self):
        try:
            yield
        finally:
            self.in_progress -= 1
            self.log.debug("done", in_progress=self.in_progress)
            if not self.in_progress:
                self.log.debug("idling")
                self.idle.set()

    async def wait(self):
        await self.idle.wait()
        return self


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
    models: dict[str, AIModel]
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
    async def monitor_health_and_update_models(self, pool): ...


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
            self.log("failed loading model", exception=e, model=model_id)
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

    async def monitor_health_and_update_models(self, pool):
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

    # XXX fail over to next ?
    #                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #   File "/Users/ctheune/Code/skvaider/.venv/lib/python3.11/site-packages/httpx/_client.py", line 1730, in _send_single_request
    #     response = await transport.handle_async_request(request)
    #                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #   File "/Users/ctheune/Code/skvaider/.venv/lib/python3.11/site-packages/httpx/_transports/default.py", line 393, in handle_async_request
    #     with map_httpcore_exceptions():
    #   File "/Users/ctheune/.nix-profile/lib/python3.11/contextlib.py", line 158, in __exit__
    #     self.gen.throw(typ, value, traceback)
    #   File "/Users/ctheune/Code/skvaider/.venv/lib/python3.11/site-packages/httpx/_transports/default.py", line 118, in map_httpcore_exceptions
    #     raise mapped_exc(message) from exc
    # httpx.ConnectError: All connection attempts failed


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

    async def monitor_health_and_update_models(self, pool):
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


class ProxyRequest:
    backend_available: asyncio.Event
    model: AIModel = None

    def __init__(self):
        self.backend_available = asyncio.Event()


class Pool:
    backends: list["Backend"]
    health_check_tasks: list[asyncio.Task]
    queues: dict[str, asyncio.Queue]  # one queue per model

    def __init__(self):
        self.backends = []
        self.health_check_tasks = []
        self.queues = {}
        self.models = {}
        self.queue_tasks = {}

    def add_backend(self, backend):
        self.backends.append(backend)
        self.health_check_tasks.append(
            utils.create_task(backend.monitor_health_and_update_models(self))
        )

    def update_model_maps(self):
        # XXX the same model must not be owned by different organizations!
        # This requires a bit more thought how to handle consistency if
        # backends answer with conflicting/differing model data.
        self.models.clear()
        for backend in self.backends:
            self.models.update(backend.models)

        # Add new models
        for model_id in self.models:
            if model_id in self.queues:
                continue
            self.queues[model_id] = asyncio.Queue()
            self.queue_tasks[model_id] = utils.create_task(
                self.assign_backends(model_id)
            )

        # Remove outdated model queues and tasks
        for model_id, task in self.queue_tasks.items():
            if model_id in self.models:
                continue
            task.cancel()
            del self.queue_tasks[model_id]

        for model_id in self.queues:
            if model_id in self.models:
                continue
            del self.queues[model_id]

    async def assign_backends(self, model_id: str):
        """Continuously assign requests to backends.

        Perform batching and model distribution and warmup.

        """
        while True:
            log.debug("waiting for request", model=model_id)
            queue = self.queues[model_id]
            request_batch = [await queue.get()]
            log.debug("got request", model=model_id)

            # Now, are there any backends with the model loaded and are they available?
            while not (
                model_backends := [
                    b for b in self.backends if model_id in b.models
                ]
            ):
                log.warning("no backends with model available", model=model_id)
                await asyncio.sleep(1)

            loaded_backends = [
                b for b in model_backends if b.models[model_id].is_loaded
            ]
            idle_backends = [
                b for b in loaded_backends if b.models[model_id].idle.is_set()
            ]
            not_loaded_backends = [
                b for b in model_backends if not b.models[model_id].is_loaded
            ]

            if (
                not idle_backends
                and len(loaded_backends) < 2
                and not_loaded_backends
            ):  # At most 2 instances per model
                # Load the model on a host with as little used memory as possible
                # if we have spare hosts.
                not_loaded_backends.sort(key=lambda b: b.memory_usage)
                new_backend = not_loaded_backends[0]
                log.debug(
                    "warming up model on new backend",
                    backend=new_backend.url,
                    model=model_id,
                )
                await new_backend.load_model_with_options(model_id)
                idle_backends.insert(0, new_backend)

            if not idle_backends:
                # Need to wait for an idle backend
                log.debug("waiting for idle backends", model=model_id)
                backends_to_wait_for = [
                    utils.create_task(b.models[model_id].wait())
                    for b in model_backends
                ]
                idle_backends, _ = await asyncio.wait(
                    backends_to_wait_for,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                # the above is a set, we want a list
                idle_backends = [b for b in idle_backends]
            backend = idle_backends[0]
            model = backend.models[model_id]
            log.debug("got idle backend", backend=backend.url, model=model_id)

            # This should not be necessary, but it should also be gratuitous.

            await backend.load_model_with_options(model_id)
            log.debug("gathering more batchable requests", model=model_id)
            # Prime the model
            # Wait up to 0.1s for up to N requests
            more_request_tasks = await asyncio.gather(
                *[
                    asyncio.wait_for(queue.get(), 0.001)
                    for _ in range(model.limit - 1)
                ],
                return_exceptions=True,
            )
            request_batch.extend(
                [t for t in more_request_tasks if not isinstance(t, Exception)]
            )
            for request in request_batch:
                log.debug(
                    "assigning request to backend",
                    model=model_id,
                    backend=backend.url,
                )
                model.in_progress += 1
                request.model = model
                request.backend_available.set()
            model.idle.clear()

    def close(self):
        for task in self.health_check_tasks:
            task.cancel()
        for task in self.queue_tasks.values():
            task.cancel()

    # def choose_backend(self, model_id: str):
    #     """Return a list of all healthy connections sorted by least number of
    #     current connections.
    #     """
    #     healthy = filter(lambda b: b.healthy, self.backends)
    #     with_model = filter(lambda b: model_ in b.models, healthy)
    #     available_models = sorted(with_model, key=lambda x: x.connections)
    #     if not available_models:
    #         raise HTTPException(
    #             400,
    #             f"The model: `{model_id}` does not exist",
    #         )

    #     ranked_models = sorted(
    #         with_model,
    #         key=lambda x: (
    #             x.models[model_id].is_loaded,
    #             x.models[model_id].open_slots,
    #         ),
    #     )

    #     # backend = self.choose_backend(model_id)
    #     # model = backend.models[model_id]
    #     # model.last_used = time.time()
    #     # try:
    #     #     yield backend
    #     # finally:
    #     #     model.in_progress
    #     #     model.connections -= 1

    #     # - consider actual ram usage on backends before asking a server
    #     #   to load a fresh model

    #     # Better decision for later
    #     # - perform batching on our side and then unblock a number of requests at the same time
    #     # - do not select a backend until it has capacity for us

    #     return ranked[_models0]

    @contextlib.asynccontextmanager
    async def use(self, model_id: str):
        request = ProxyRequest()
        assert model_id in self.queues
        log.debug("queuing request", model=model_id)
        queue = self.queues[model_id]
        await queue.put(request)
        log.debug("waiting for backend to become available", model=model_id)
        await request.backend_available.wait()
        log.debug(
            "got backend", backend=request.model.backend.url, model=model_id
        )
        async with request.model.use():
            yield request.model.backend


class OpenAIProxy:
    """Intermediate the proxy logic between FastAPI and the OpenAI API-compatible backends."""

    def __init__(self, services: svcs.fastapi.DepContainer):
        self.services = services
        self.pool = self.services.get(Pool)

    async def proxy(self, request, endpoint, allow_stream=True):
        request_data = await request.json()
        request_data["store"] = False
        request.state.model = request_data["model"]
        request.state.stream = allow_stream and request_data.get(
            "stream", False
        )

        if request.state.model not in self.pool.queues:
            raise HTTPException(
                400,
                f"The model `{request.state.model}` is currently not available.",
            )

        if request.state.stream:
            # We need to place the context manager in a scope that is valid while the response is
            # streaming, so wrap the original streaming method and iterate there
            async def stream(stream_aws, context):
                try:
                    async for chunk in stream_aws:
                        yield chunk
                finally:
                    await context.__aexit__(None, None, None)

            context = self.pool.use(request.state.model)
            backend = await context.__aenter__()
            request.state.backend = backend
            stream_aws = backend.post_stream(endpoint, request_data)
            return StreamingResponse(
                stream(stream_aws, context),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            async with self.pool.use(request.state.model) as backend:
                return await backend.post(endpoint, request_data)


class ListResponse(BaseModel, Generic[T]):
    object: str = "list"
    data: list[T]


@router.get("/v1/models")
async def list_models(
    services: svcs.fastapi.DepContainer,
) -> ListResponse[AIModel]:
    pool = services.get(Pool)
    return ListResponse[AIModel](data=pool.models.values())


@router.get("/v1/models/{model_id}")
async def get_model(
    model_id: str, services: svcs.fastapi.DepContainer
) -> AIModel:
    pool = services.get(Pool)
    return pool.models[model_id]


@router.post("/v1/chat/completions")
async def chat_completions(
    r: Request, services: svcs.fastapi.DepContainer
) -> Any:
    proxy = OpenAIProxy(services)
    return await proxy.proxy(r, "/v1/chat/completions")


@router.post("/v1/completions")
async def completions(r: Request, services: svcs.fastapi.DepContainer) -> Any:
    proxy = OpenAIProxy(services)
    return await proxy.proxy(r, "/v1/completions")


@router.post("/v1/embeddings")
async def embeddings(r: Request, services: svcs.fastapi.DepContainer) -> Any:
    proxy = OpenAIProxy(services)
    return await proxy.proxy(r, "/v1/embeddings", allow_stream=False)
