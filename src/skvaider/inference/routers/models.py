import json
from pathlib import Path
from typing import Any

import anyio
import httpx
import structlog
import svcs
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from skvaider.inference.manager import Manager

router = APIRouter()
log = structlog.get_logger()


class DownloadRequest(BaseModel):
    url: str
    model_name: str
    metadata: dict[str, Any] | None = None


@router.post("/get_running_model_or_load")
async def load_model(
    request: Request,
    services: svcs.fastapi.DepContainer,
):
    manager = services.get(Manager)

    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    model_name = body.get("model")
    if not model_name:
        raise HTTPException(status_code=400, detail="Model not specified")

    try:
        running_model = await manager.get_or_start_model(model_name)
    except Exception as e:
        log.error("Failed to start model", model=model_name, error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to start model: {e}"
        )

    if not running_model:
        raise HTTPException(status_code=404, detail="Model not found")

    return {"endpoint": running_model.endpoint}


@router.post("/unload")
async def unload_model(
    request: Request,
    services: svcs.fastapi.DepContainer,
):
    manager = services.get(Manager)

    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    model_name = body.get("model")
    if not model_name:
        raise HTTPException(status_code=400, detail="Model not specified")

    await manager.unload_model(model_name)
    return {"status": "ok"}


@router.get("/models")
async def list_models(
    services: svcs.fastapi.DepContainer,
):
    manager = services.get(Manager)

    return {"models": await manager.list_models()}


@router.get("/running_models")
async def list_running_models(
    services: svcs.fastapi.DepContainer,
):
    manager = services.get(Manager)
    return {
        "models": list(
            m.config.id for m in manager.models.values() if m.running
        )
    }


@router.api_route(
    "/model/{model_name}/proxy/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE"],
)
async def proxy_request(
    model_name: str,
    path: str,
    request: Request,
    services: svcs.fastapi.DepContainer,
):
    manager = services.get(Manager)

    model = await manager.get_or_start_model(model_name)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    url = f"{model.endpoint}/{path}"
    if request.query_params:
        url += f"?{request.query_params}"

    client = httpx.AsyncClient()

    # Exclude headers that might cause issues
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)

    req = client.build_request(
        request.method,
        url,
        headers=headers,
        content=await request.body(),
    )

    try:
        rp = await client.send(req, stream=True)
    except Exception as e:
        await client.aclose()
        raise HTTPException(status_code=500, detail=f"Proxy error: {e}")

    async def streaming_content():
        try:
            async for chunk in rp.aiter_raw():
                yield chunk
        finally:
            await rp.aclose()
            await client.aclose()

    return StreamingResponse(
        streaming_content(),
        status_code=rp.status_code,
        headers=dict(rp.headers),
    )
