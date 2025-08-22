import asyncio
import contextlib
from typing import Any, AsyncGenerator, Generic, TypeVar

import httpx
import svcs
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()

T = TypeVar("T")


class AIModel(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str


class Backend:
    url: str
    connections: int = 0
    health_interval: int = 15
    healthy: bool = False
    unhealthy_reason: str = ""
    models: dict[str, AIModel]

    def __init__(self, url):
        self.url = url
        self.models = {}

    async def post(self, path: str, data: dict):
        async with httpx.AsyncClient() as client:
            r = await client.post(self.url + path, json=data, timeout=120)
            return r.json()

    async def post_stream(
        self, path: str, data: dict
    ) -> AsyncGenerator[str, None]:
        """Stream responses from the backend"""
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST", self.url + path, json=data, timeout=120
            ) as response:
                async for chunk in response.aiter_text():
                    if chunk.strip():
                        yield chunk

    async def monitor_health_and_update_models(self):
        while True:
            try:
                async with httpx.AsyncClient() as client:
                    r = await client.get(self.url + "/v1/models")
                self.models.clear()
                new_models = r.json()["data"] or ()
                for model in new_models:
                    self.models[model["id"]] = AIModel(
                        id=model["id"],
                        created=model["created"],
                        owned_by=model["owned_by"],
                    )
            except Exception as e:
                if self.healthy:
                    print(f"Marking {self.url} as UNHEALTHY: {e}")
                self.healthy = False
                self.unhealthy_reason = str(e)
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


class Pool:
    backends: list["Backend"]
    health_check_tasks: list[asyncio.Task]

    def __init__(self):
        self.backends = []
        self.health_check_tasks = []

    def models(self):
        # XXX the same model must not be owned by different organizations!
        # This requires a bit more thought how to handle consistency if
        # backends answer with conflicting/differing model data.
        models = {}
        for backend in self.backends:
            models.update(backend.models)
        return models

    def add_backend(self, backend):
        self.backends.append(backend)
        self.health_check_tasks.append(
            asyncio.create_task(backend.monitor_health_and_update_models())
        )

    def close(self):
        for task in self.health_check_tasks:
            task.cancel()

    def choose_backend(self, model):
        """Return a list of all healthy connections sorted by least number of
        current connections.
        """
        healthy = filter(lambda b: b.healthy, self.backends)
        with_model = filter(lambda b: model in b.models, healthy)
        return sorted(with_model, key=lambda x: x.connections)[-1]

    @contextlib.contextmanager
    def use(self, model):
        backend = self.choose_backend(model)
        backend.connections += 1
        try:
            yield backend
        finally:
            backend.connections -= 1


class ListResponse(BaseModel, Generic[T]):
    object: str = "list"
    data: list[T]


@router.get("/v1/models")
async def list_models(
    services: svcs.fastapi.DepContainer,
) -> ListResponse[AIModel]:
    pool = services.get(Pool)
    return ListResponse[AIModel](data=pool.models().values())


@router.get("/v1/models/{model_id}")
async def get_model(
    model_id: str, services: svcs.fastapi.DepContainer
) -> AIModel:
    pool = services.get(Pool)
    return pool.models()[model_id]


@router.post("/v1/chat/completions")
async def chat_completions(
    r: Request, services: svcs.fastapi.DepContainer
) -> Any:
    request_data = await r.json()
    request_data["store"] = False
    model = request_data["model"]

    pool = services.get(Pool)

    with pool.use(model) as backend:
        # XXX pass through headers?
        # Check if streaming is requested
        stream = request_data.get("stream", False)

        if stream:
            # Return streaming response
            async def generate():
                async for chunk in backend.post_stream(
                    "/v1/chat/completions", request_data
                ):
                    yield chunk

            return StreamingResponse(
                generate(),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            # Return regular JSON response
            return await backend.post("/v1/chat/completions", request_data)


@router.post("/v1/completions")
async def completions(r: Request, services: svcs.fastapi.DepContainer) -> Any:
    request_data = await r.json()
    request_data["store"] = False
    model = request_data["model"]

    pool = services.get(Pool)

    with pool.use(model) as backend:
        # Check if streaming is requested
        stream = request_data.get("stream", False)

        if stream:
            # Return streaming response
            async def generate():
                async for chunk in backend.post_stream(
                    "/v1/completions", request_data
                ):
                    yield chunk

            return StreamingResponse(
                generate(),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            # Return regular JSON response
            return await backend.post("/v1/completions", request_data)
