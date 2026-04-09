import json
import time
from collections.abc import AsyncGenerator
from typing import cast

import httpx
import structlog
import svcs
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from skvaider.inference import metrics
from skvaider.inference.manager import Manager, ModelAlreadyLoading
from skvaider.inference.model import Model
from skvaider.typing import JSONObject

router = APIRouter()
log = structlog.get_logger()

# Endpoints where we can extract token usage from the response body.
TOKEN_ENDPOINTS = {"v1/completions", "v1/chat/completions"}


def model_info(model: Model, manager: Manager) -> JSONObject:
    return {
        "id": model.config.id,
        "status": list(model.status),
        "max_requests": model.config.max_requests,
        "memory_usage": {
            monitor.id: monitor.model_usage(model)
            for monitor in manager.monitors.values()
        },
    }


def _extract_token_usage(body: bytes, model_name: str) -> None:
    """Best-effort extraction of token usage from a completed response.

    Supports both plain JSON responses and SSE streams. In SSE streams
    the usage object is typically in the last ``data:`` line before
    ``data: [DONE]``.
    """
    text = body.decode("utf-8", errors="replace")

    # Try plain JSON first (non-streaming response).
    try:
        data = json.loads(text)
        _record_usage(data, model_name)
        return
    except (json.JSONDecodeError, ValueError):
        pass

    # Fall back to SSE: scan backwards for the last data line with usage.
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            data = json.loads(line[6:])
            if "usage" in data:
                _record_usage(data, model_name)
                return
        except (json.JSONDecodeError, ValueError):
            continue


def _record_usage(data: dict[str, object], model_name: str) -> None:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return
    usage_typed = cast(dict[str, object], usage)
    prompt = usage_typed.get("prompt_tokens", 0)
    completion = usage_typed.get("completion_tokens", 0)
    if isinstance(prompt, (int, float)) and prompt:
        metrics.inference_tokens_prompt.labels(model=model_name).inc(prompt)
    if isinstance(completion, (int, float)) and completion:
        metrics.inference_tokens_generated.labels(model=model_name).inc(
            completion
        )


# -- Routes ------------------------------------------------------------------


@router.get("/models/{model_name}")
async def get_model_info(
    model_name: str,
    services: svcs.fastapi.DepContainer,
) -> JSONObject:
    manager = services.get(Manager)
    model_name = model_name.lower()

    model = manager.models.get(model_name)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # keep in sync with list_models
    return model_info(model, manager)


@router.post("/models/{model_name}/load")
async def load_model(
    model_name: str,
    request: Request,
    services: svcs.fastapi.DepContainer,
) -> JSONObject:
    manager = services.get(Manager)
    model_name = model_name.lower()

    if not model_name:
        raise HTTPException(status_code=400, detail="Model not specified")

    try:
        running_model = await manager.start_model(model_name)

    except ModelAlreadyLoading:
        raise HTTPException(
            status_code=409, detail="Model is already loading/unloading"
        )
    except Exception as e:
        log.exception("Failed to start model", model=model_name, error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to start model: {e}"
        )

    if not running_model:
        raise HTTPException(status_code=404, detail="Model not found")

    return {"endpoint": running_model.endpoint}


@router.post("/models/{model_name}/unload")
async def unload_model(
    model_name: str,
    services: svcs.fastapi.DepContainer,
) -> JSONObject:
    manager = services.get(Manager)
    model_name = model_name.lower()

    if not model_name:
        raise HTTPException(status_code=400, detail="Model not specified")

    await manager.unload_model(model_name)
    return {"status": "ok"}


@router.get("/models")
async def list_models(
    services: svcs.fastapi.DepContainer,
) -> JSONObject:
    manager = services.get(Manager)
    return {"models": [model_info(m, manager) for m in manager.list_models()]}


@router.api_route(
    "/models/{model_name}/proxy/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE"],
)
async def proxy_request(
    model_name: str,
    path: str,
    request: Request,
    services: svcs.fastapi.DepContainer,
) -> StreamingResponse:
    manager = services.get(Manager)
    model_name = model_name.lower()
    endpoint = f"/proxy/{path}"
    start_time = time.monotonic()

    model = await manager.use_model(model_name)
    if not model:
        metrics.inference_requests_total.labels(
            model=model_name, endpoint=endpoint, status="unavailable"
        ).inc()
        raise HTTPException(status_code=540, detail="Model unavailable")

    await model.lock.user_acquire()
    try:
        url = f"{model.endpoint}/{path}"
        if request.query_params:
            url += f"?{request.query_params}"

        client = httpx.AsyncClient(
            timeout=300
        )  # XXX make configurable and/or recommend streaming?

        proxy_header_names = ["Content-Type"]
        proxy_headers: dict[str, str] = {}
        for candidate in proxy_header_names:
            if candidate not in request.headers:
                continue
            proxy_headers[candidate] = request.headers[candidate]

        req = client.build_request(
            request.method,
            url,
            content=await request.body(),
            headers=proxy_headers,
        )

        try:
            upstream = await client.send(req, stream=True)
        except Exception:
            await client.aclose()
            metrics.inference_requests_total.labels(
                model=model_name, endpoint=endpoint, status="error"
            ).inc()
            log.exception("Error proxying to model", model=model_name)
            raise HTTPException(
                status_code=500,
                detail=f"Error in model '{model_name}' request",
            )

        collect_tokens = path in TOKEN_ENDPOINTS
        chunks: list[bytes] = []

        async def stream() -> AsyncGenerator[bytes]:
            try:
                async for chunk in upstream.aiter_raw():
                    if collect_tokens:
                        chunks.append(chunk)
                    yield chunk
            finally:
                await upstream.aclose()
                await client.aclose()
                await model.lock.user_release()

                duration = time.monotonic() - start_time
                metrics.inference_requests_total.labels(
                    model=model_name, endpoint=endpoint, status="success"
                ).inc()
                metrics.inference_request_duration_seconds.labels(
                    model=model_name, endpoint=endpoint
                ).observe(duration)

                if collect_tokens and chunks:
                    _extract_token_usage(b"".join(chunks), model_name)

        return StreamingResponse(
            stream(),
            status_code=upstream.status_code,
            headers=dict(upstream.headers),
        )
    except BaseException:
        # If we never started streaming, release the lock here.
        await model.lock.user_release()
        raise
