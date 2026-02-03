from collections.abc import AsyncGenerator

import httpx
import structlog
import svcs
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from skvaider.inference.manager import Manager, Model
from skvaider.typing import JSONObject

router = APIRouter()
log = structlog.get_logger()


def model_info(model: Model, manager: Manager) -> JSONObject:
    return {
        "id": model.config.id,
        "status": list(model.status),
        "memory_usage": {
            monitor.id: monitor.model_usage(model)
            for monitor in manager.monitors.values()
        },
    }


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
    except Exception as e:
        log.error("Failed to start model", model=model_name, error=str(e))
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

    model = await manager.use_model(model_name)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    await model.lock.user_acquire()
    try:
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
    finally:
        await model.lock.user_release()
