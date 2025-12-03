"""Open-AI compatible API based on Ollama.

This uses Ollama-internal APIs for better load-balancing but exposes a pure OpenAI-compatible API.

"""

from typing import Any

import structlog
import svcs
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from skvaider.proxy.models import AIModel, ListResponse
from skvaider.proxy.pool import Pool

router = APIRouter()

log = structlog.stdlib.get_logger()


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
