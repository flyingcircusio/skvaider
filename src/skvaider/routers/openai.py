"""Open-AI compatible API."""

from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager
from typing import Any, Generic, TypeVar

import structlog
import svcs
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from skvaider.proxy.backends import Backend
from skvaider.proxy.models import AIModel
from skvaider.proxy.pool import Pool

T = TypeVar("T")


class ListResponse(BaseModel, Generic[T]):
    object: str = "list"
    data: list[T]


router = APIRouter()

log = structlog.stdlib.get_logger()


class OpenAIProxy:
    """Intermediate the proxy logic between FastAPI and the OpenAI API-compatible backends."""

    def __init__(self, services: svcs.fastapi.DepContainer):
        self.services = services
        self.pool = self.services.get(Pool)

    async def proxy(
        self,
        request: Request,
        endpoint: str,
        allow_stream: bool = True,
    ) -> StreamingResponse | Any:
        request_data: dict[str, Any] = await request.json()
        request_data["store"] = False
        request_data["model"] = request_data["model"].lower()
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
            async def stream(
                stream_aws: AsyncGenerator[str],
                context: AbstractAsyncContextManager[Backend],
            ) -> AsyncGenerator[str]:
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


@router.get("/v1/models")
async def list_models(
    services: svcs.fastapi.DepContainer,
) -> ListResponse[AIModel]:
    pool = services.get(Pool)
    models: dict[str, AIModel] = {}
    # The pool used to keep track of the pydantic models but I didn't like
    # that we had objects there where we only needed the ids. Here we do
    # need the objects, so we need to sample them.
    for backend in pool.backends:
        models.update(backend.models)
    return ListResponse[AIModel](data=list(models.values()))


@router.get("/v1/models/{model_id}")
async def get_model(
    model_id: str, services: svcs.fastapi.DepContainer
) -> AIModel:
    model_id = model_id.lower()
    pool = services.get(Pool)
    # See list_models
    for backend in pool.backends:
        if model_id in backend.models:
            return backend.models[model_id]
    raise HTTPException(
        404,
        f"Unkonwn model `{model_id}`.",
    )


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
