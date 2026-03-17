"""Open-AI compatible API."""

import time
from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager
from typing import Any, Generic, TypeVar

import structlog
import svcs
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from skvaider import metrics
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

    def __init__(
        self, services: svcs.fastapi.DepContainer, max_retries: int = 3
    ):
        self.services = services
        self.pool = self.services.get(Pool)
        self.max_retries = max_retries

    async def _execute_with_retry(
        self,
        request: Request,
        endpoint: str,
        data: dict[str, Any],
    ) -> Any:
        excluded_backends: set[str] = set()
        semaphore = self.pool.semaphores[request.state.model]

        for attempt in range(self.max_retries):
            async with semaphore.use(
                excluded_backends=excluded_backends
            ) as backend:
                if not backend:
                    break
                excluded_backends.add(backend.url)
                try:
                    result = await backend.post(endpoint, data)
                    metrics.gateway_backend_requests_total.labels(
                        model=request.state.model,
                        backend=backend.url,
                        status="success",
                        endpoint=endpoint,
                        streaming=False,
                    ).inc()
                    return result
                except HTTPException as e:
                    metrics.gateway_backend_requests_total.labels(
                        backend=backend.url,
                        model=request.state.model,
                        status="error",
                        endpoint=endpoint,
                        streaming=False,
                    ).inc()
                    if e.status_code == 540:  # Backend currently unavailable
                        metrics.gateway_backend_retry_total.labels(
                            model=request.state.model,
                            backend=backend.url,
                            endpoint=endpoint,
                            streaming=False,
                            reason="backend_unavailable",
                        ).inc()
                        log.warning(
                            "Backend unavailable, retrying",
                            model=request.state.model,
                            attempt=attempt + 1,
                            excluded_backends=excluded_backends,
                        )

        raise HTTPException(status_code=503, detail="Service unavailable")

    async def _execute_stream_with_retry(
        self,
        request: Request,
        endpoint: str,
        data: dict[str, Any],
    ) -> StreamingResponse:
        excluded_backends: set[str] = set()
        semaphore = self.pool.semaphores[request.state.model]

        for attempt in range(self.max_retries):
            context = semaphore.use(excluded_backends=excluded_backends)
            backend = await context.__aenter__()
            if backend is None:
                break
            excluded_backends.add(backend.url)
            try:
                stream_aws = backend.post_stream(endpoint, data)
                first_chunk = await anext(stream_aws)
                metrics.gateway_backend_requests_total.labels(
                    backend=backend.url,
                    endpoint=endpoint,
                    model=request.state.model,
                    status="success",
                    streaming=True,
                ).inc()
            except StopAsyncIteration:
                await context.__aexit__(None, None, None)
                # Track backend request failure
                if backend:
                    metrics.gateway_backend_requests_total.labels(
                        backend=backend.url,
                        endpoint=endpoint,
                        model=request.state.model,
                        status="error",
                        streaming=True,
                    ).inc()
                # Empty stream
                return StreamingResponse(
                    iter([]),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    },
                )
            except HTTPException as e:
                await context.__aexit__(None, None, None)
                # Track backend request failure
                if backend:
                    metrics.gateway_backend_requests_total.labels(
                        backend=backend.url,
                        endpoint=endpoint,
                        model=request.state.model,
                        status="error",
                        streaming=True,
                    ).inc()
                if e.status_code == 540:
                    metrics.gateway_backend_retry_total.labels(
                        backend=backend.url,
                        endpoint=endpoint,
                        model=request.state.model,
                        reason="backend_unavailable",
                        streaming=True,
                    ).inc()
                    log.warning(
                        "Backend unavailable, retrying",
                        model=request.state.model,
                        attempt=attempt + 1,
                        excluded_backends=excluded_backends,
                    )
                continue

            # Success
            async def stream(
                first: str,
                stream_aws: AsyncGenerator[str],
                context: AbstractAsyncContextManager[Backend | None],
            ) -> AsyncGenerator[str]:
                try:
                    yield first
                    async for chunk in stream_aws:
                        yield chunk
                finally:
                    await context.__aexit__(None, None, None)

            request.state.backend = backend
            return StreamingResponse(
                stream(first_chunk, stream_aws, context),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        raise HTTPException(status_code=503, detail="Service unavailable")

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

        if request.state.model not in self.pool.model_configs:
            raise HTTPException(
                400,
                f"The model `{request.state.model}` is not known.",
            )

        # Track active requests
        metrics.gateway_active_requests.labels(
            model=request.state.model,
            endpoint=endpoint,
            streaming=request.state.stream,
        ).inc()
        start_time = time.time()
        status = "success"

        try:
            if request.state.stream:
                result = await self._execute_stream_with_retry(
                    request, endpoint, request_data
                )
            else:
                result = await self._execute_with_retry(
                    request,
                    endpoint,
                    request_data,
                )

            return result
        except Exception:
            status = "error"
            raise
        finally:
            # Track request completion
            metrics.gateway_active_requests.labels(
                model=request.state.model,
                endpoint=endpoint,
                streaming=request.state.stream,
            ).dec()
            duration = time.time() - start_time
            metrics.gateway_request_duration_seconds.labels(
                model=request.state.model,
                endpoint=endpoint,
                streaming=request.state.stream,
            ).observe(duration)
            metrics.gateway_requests_total.labels(
                model=request.state.model,
                endpoint=endpoint,
                status=status,
                streaming=request.state.stream,
            ).inc()


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
        f"Unknown model `{model_id}`.",
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
