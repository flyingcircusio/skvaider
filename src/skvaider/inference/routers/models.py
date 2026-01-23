from collections.abc import AsyncGenerator

import httpx
import structlog
import svcs
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from skvaider.inference.manager import Manager
from skvaider.typing import JSONObject

router = APIRouter()
log = structlog.get_logger()


@router.get("/models/{model_name}")
async def get_model_info(
    model_name: str,
    services: svcs.fastapi.DepContainer,
) -> JSONObject:
    manager = services.get(Manager)

    model = manager.models.get(model_name)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    return {
        "id": model.config.id,
        "status": model.status,
        "endpoint": model.endpoint,
        "healthy": model.is_healthy,
    }


@router.post("/models/{model_name}/load")
async def load_model(
    model_name: str,
    request: Request,
    services: svcs.fastapi.DepContainer,
) -> JSONObject:
    manager = services.get(Manager)

    if not model_name:
        raise HTTPException(status_code=400, detail="Model not specified")

    try:
        running_model = await manager.get_or_start_model(model_name)
    except Exception as e:
        log.exception("Failed to start model", model=model_name)
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

    if not model_name:
        raise HTTPException(status_code=400, detail="Model not specified")

    await manager.unload_model(model_name)
    return {"status": "ok"}


@router.get("/models")
async def list_models(
    services: svcs.fastapi.DepContainer,
) -> JSONObject:
    manager = services.get(Manager)

    return {
        "models": await manager.list_models(),
        "running": [m.config.id for m in manager.models.values() if m.running],
    }


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

    model = await manager.get_or_start_model(model_name)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    url = f"{model.endpoint}/{path}"
    if request.query_params:
        url += f"?{request.query_params}"

    client = httpx.AsyncClient(timeout=60)  # LLM requests may take time

    req = client.build_request(
        request.method,
        url,
        content=await request.body(),
    )

    try:
        rp = await client.send(req, stream=True)
    except Exception:
        await client.aclose()

        log.exception(
            "Error in model request {model_name}", model_name=model_name
        )
        raise HTTPException(
            status_code=500, detail=f"Error in model '{model_name}' request"
        )

    async def streaming_content() -> AsyncGenerator[bytes]:
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
