"""Open-AI compatible API based on Ollama.

This uses Ollama-internal APIs for better load-balancing but exposes a pure OpenAI-compatible API.

"""

import asyncio
import contextlib
import logging
from typing import Any, AsyncGenerator, Dict, Generic, Optional, TypeVar

import httpx
import svcs
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter()

T = TypeVar("T")


def log_task_exception(task: asyncio.Task) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        pass  # Task cancellation should not be logged as an error.
    except Exception:  # pylint: disable=broad-except
        logging.exception("Exception raised by task = %r", task)


def create_logged_task(aw):
    t = asyncio.create_task(aw)
    t.add_done_callback(log_task_exception)
    return t


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

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.idle = asyncio.Event()
        self.idle.set()

    @contextlib.asynccontextmanager
    async def use(self):
        try:
            yield
        finally:
            self.in_progress -= 1
            if not self.in_progress:
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


class Backend:
    """Connection to a single backend."""

    url: str

    health_interval: int = 15
    healthy: bool = False
    unhealthy_reason: str = ""
    models: dict[str, AIModel]
    model_config: ModelConfig

    def __init__(self, url, model_config):
        self.url = url
        self.models = {}
        self.model_config = model_config

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
        except Exception as e:
            print(f"Failed to load model {model_id} with options: {e}")
            raise

    async def monitor_health_and_update_models(self, pool):
        print(f"Starting monitor for {self.url}")
        while True:
            try:
                async with httpx.AsyncClient(follow_redirects=True) as client:
                    r = await client.get(self.url + "/v1/models")
                    known_models = r.json()["data"] or ()
                async with httpx.AsyncClient(follow_redirects=True) as client:
                    r = await client.get(self.url + "/api/ps")
                    loaded_models = set([m["name"] for m in r.json()["models"]])

                current_models = self.models

                updated_models = {}
                for model in known_models:
                    if model["id"] not in known_models:
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

                    model_obj.is_loaded = model_obj.id in loaded_models

                    updated_models[model_obj.id] = model_obj

                self.models = updated_models

                pool.update_model_maps()

            except Exception as e:
                if self.healthy:
                    print(f"Marking {self.url} as UNHEALTHY: {e}")
                self.healthy = False
                self.unhealthy_reason = str(e)
                # Reset our model knowledge, drop statistics
                self.models = {}
            else:
                if not self.healthy:
                    print(f"Marking {self.url} as HEALTHY.")
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
            create_logged_task(backend.monitor_health_and_update_models(self))
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
            self.queue_tasks[model_id] = create_logged_task(
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
        while True:
            print(f"{model_id}: waiting for idle backends")
            backends_to_wait_for = [
                create_logged_task(b.models[model_id].wait())
                for b in self.backends
                if model_id in b.models
            ]
            if not backends_to_wait_for:
                print(f"{model_id} - no backends ready")
                await asyncio.sleep(1)
                continue
            idle_backends, _ = await asyncio.wait(
                backends_to_wait_for,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in idle_backends:
                model = task.result()
                backend = model.backend
                print(f"{model_id} got idle backend {backend.url}")
                queue = self.queues[model_id]
                print(f"{model_id}: assigning tasks to backend {backend.url}")
                # Wait infinitely long for the first request
                request_batch = [await queue.get()]
                print(f"{model_id}: priming")
                await backend.load_model_with_options(model_id)
                print(f"{model_id}: gathering batchable requests")
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
                    [
                        t
                        for t in more_request_tasks
                        if not isinstance(t, Exception)
                    ]
                )
                for request in request_batch:
                    print(
                        f"{model}: assigning request to backend {backend.url}"
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
        if model_id not in self.queues:
            raise HTTPException(
                400,
                f"The model `{model_id}` is currently not available.",
            )
        print(f"queuing request for {model_id}")
        queue = self.queues[model_id]
        await queue.put(request)
        print("waiting for backend to become available")
        await request.backend_available.wait()
        print("got backend")
        async with request.model.use():
            print("making backend available")
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

        async with self.pool.use(request.state.model) as backend:
            if request.state.stream:
                return StreamingResponse(
                    backend.post_stream(endpoint, request_data),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    },
                )
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
