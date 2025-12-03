import json
from pathlib import Path
from typing import Any

import anyio
import httpx
import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from skvaider.inference.state import manager

router = APIRouter()
log = structlog.get_logger()


class DownloadRequest(BaseModel):
    url: str
    filename: str
    metadata: dict[str, Any] | None = None


@router.post("/load")
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
    models_dir = Path("models")
    try:
        models_dir.mkdir(exist_ok=True)
    except OSError as e:
        log.error("Failed to create models directory", error=str(e))
        raise HTTPException(
            status_code=500, detail="Could not create models directory"
        )

    file_path = models_dir / request.filename

    # Basic security check to prevent directory traversal
    if not file_path.resolve().is_relative_to(models_dir.resolve()):
        raise HTTPException(status_code=400, detail="Invalid filename")

    if request.metadata:
        metadata_path = models_dir / (request.filename + ".json")
        try:
            async with await anyio.open_file(metadata_path, "w") as f:
                await f.write(json.dumps(request.metadata))
        except OSError as e:
            log.error("Failed to write metadata", error=str(e))
            # We continue even if metadata fails? Or fail?
            # Let's fail for now as it seems important.
            raise HTTPException(
                status_code=500, detail="Failed to write metadata"
            )

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
