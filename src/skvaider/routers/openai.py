"""Open-AI compatible API."""

import json
import time
from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager
from typing import Any, Generic, TypeVar

import structlog
import svcs
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
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


def _extract_usage(request: Any, usage: dict[str, Any] | None) -> None:
    if not usage:
        return
    request.state.tokens_prompt = usage.get("prompt_tokens", 0)
    request.state.tokens_completion = usage.get("completion_tokens", 0)
    request.state.tokens_total = usage.get("total_tokens", 0)


def _parse_sse_usage(chunk: str) -> dict[str, Any] | None:
    # Chunks from the backend are properly-assembled SSE events in "data: <json>\n\n" form.
    if not chunk.startswith("data: ") or chunk.startswith("data: [DONE]"):
        return None
    try:
        event = json.loads(chunk[6:].strip())
    except Exception:
        return None
    return event.get("usage") or None


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
            if attempt > 0:
                request.state.retries += 1
            t_queue = time.perf_counter()
            async with semaphore.use(
                excluded_backends=excluded_backends
            ) as backend:
                request.state.time_queue += time.perf_counter() - t_queue
                if not backend:
                    break
                request.state.backend = backend
                excluded_backends.add(backend.url)
                try:
                    t_server = time.perf_counter()
                    result = await backend.post(
                        endpoint, data, request.state.request_id
                    )
                    request.state.time_server += time.perf_counter() - t_server
                    metrics.gateway_backend_requests_total.labels(
                        model=request.state.model,
                        backend=backend.url,
                        status="success",
                        endpoint=endpoint,
                        streaming=False,
                    ).inc()
                    _extract_usage(request, result.get("usage"))
                    model_id = request.state.model
                    request.state.parallel_total = sum(
                        m.in_progress
                        for b in self.pool.backends
                        for m in b.models.values()
                    )
                    request.state.parallel_model = sum(
                        b.models[model_id].in_progress
                        for b in self.pool.backends
                        if model_id in b.models
                    )
                    request.state.parallel_backend = sum(
                        m.in_progress for m in backend.models.values()
                    )
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
            if attempt > 0:
                request.state.retries += 1
            t_queue = time.perf_counter()
            context = semaphore.use(excluded_backends=excluded_backends)
            backend = await context.__aenter__()
            request.state.time_queue += time.perf_counter() - t_queue
            if backend is None:
                break
            request.state.backend = backend
            excluded_backends.add(backend.url)
            try:
                stream_aws = backend.post_stream(
                    endpoint, data, request.state.request_id
                )
                t_server = time.perf_counter()
                first_chunk = await anext(stream_aws)
                request.state.time_server += time.perf_counter() - t_server
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
                if e.status_code != 540:
                    raise
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
            model_id = request.state.model
            pool = self.pool

            async def stream(
                first: str,
                stream_aws: AsyncGenerator[str],
                context: AbstractAsyncContextManager[Backend | None],
                _backend: Backend,
                _pool: Any,
                _model_id: str,
            ) -> AsyncGenerator[str]:
                try:
                    yield first
                    async for chunk in stream_aws:
                        usage = _parse_sse_usage(chunk)
                        if usage:
                            _extract_usage(request, usage)
                        yield chunk
                finally:
                    request.state.parallel_total = sum(
                        m.in_progress
                        for b in _pool.backends
                        for m in b.models.values()
                    )
                    request.state.parallel_model = sum(
                        b.models[_model_id].in_progress
                        for b in _pool.backends
                        if _model_id in b.models
                    )
                    request.state.parallel_backend = sum(
                        m.in_progress for m in _backend.models.values()
                    )
                    await context.__aexit__(None, None, None)

            request.state.backend = backend
            return StreamingResponse(
                stream(
                    first_chunk,
                    stream_aws,
                    context,
                    backend,
                    pool,
                    model_id,
                ),
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
        request.state.backend_endpoint = endpoint
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

        model_id = request.state.model

        # Track active requests
        metrics.gateway_active_requests.labels(
            model=model_id,
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
            if not isinstance(result, StreamingResponse):
                result = JSONResponse(content=result)
            return result
        except HTTPException as e:
            status = "error"
            raise HTTPException(
                status_code=e.status_code, detail=e.detail
            ) from e
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
