import json
from pathlib import Path
from typing import Any

import anyio
import httpx
import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from skvaider.inference.state import manager

router = APIRouter()
log = structlog.get_logger()


class DownloadRequest(BaseModel):
    url: str
    model_name: str
    metadata: dict[str, Any] | None = None


@router.post("/get_running_model_or_load")
async def load_model(request: Request):
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

    return {"port": running_model.port}


@router.post("/download")
async def download_model(request: DownloadRequest):
    models_dir = manager.models_dir
    try:
        models_dir.mkdir(exist_ok=True)
    except OSError as e:
        log.error("Failed to create models directory", error=str(e))
        raise HTTPException(
            status_code=500, detail="Could not create models directory"
        )

    filename = f"{request.model_name}.gguf"
    file_path = models_dir / filename

    # Basic security check to prevent directory traversal
    if not file_path.resolve().is_relative_to(models_dir.resolve()):
        raise HTTPException(status_code=400, detail="Invalid model name")

    # Enforce max one layer of hierarchy
    # We strip the models_dir prefix to check the relative path depth
    rel_path = file_path.resolve().relative_to(models_dir.resolve())
    # .gguf files are flat or one dir deep. rel_path parts include filename.
    # "model.gguf" -> 1 part. "org/model.gguf" -> 2 parts.
    if len(rel_path.parts) > 2:
        raise HTTPException(status_code=400, detail="Model hierarchy too deep")

    # Ensure parent directory exists (for nested models)
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.error("Failed to create model directory", error=str(e))
        raise HTTPException(
            status_code=500, detail="Could not create model directory"
        )

    metadata = request.metadata or {}
    metadata["name"] = request.model_name
    metadata["filename"] = filename

    metadata_path = models_dir / (request.model_name + ".json")
    try:
        async with await anyio.open_file(metadata_path, "w") as f:
            await f.write(json.dumps(metadata))
    except OSError as e:
        log.error("Failed to write metadata", error=str(e))
        # We continue even if metadata fails? Or fail?
        # Let's fail for now as it seems important.
        raise HTTPException(status_code=500, detail="Failed to write metadata")

    try:
        # Disable timeout for large downloads
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "GET", request.url, follow_redirects=True
            ) as response:
                response.raise_for_status()
                async with await anyio.open_file(file_path, "wb") as f:
                    async for chunk in response.aiter_bytes():
                        await f.write(chunk)
    except httpx.HTTPError as e:
        log.error("Download failed", error=str(e))
        raise HTTPException(status_code=502, detail="Download failed")
    except OSError as e:
        log.error("File write failed", error=str(e))
        raise HTTPException(status_code=500, detail="File write failed")

    return {"status": "downloaded", "path": str(file_path)}


@router.post("/unload")
async def unload_model(request: Request):
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
async def list_models():
    return {"models": await manager.list_models()}


@router.get("/running_models")
async def list_running_models():
    return {"models": list(manager.running_models.keys())}


@router.api_route(
    "/model/{model_name}/proxy/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE"],
)
async def proxy_request(model_name: str, path: str, request: Request):
    model = await manager.get_or_start_model(model_name)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    url = f"http://localhost:{model.port}/{path}"
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
