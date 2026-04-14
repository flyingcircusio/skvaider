import structlog
import svcs
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from skvaider.inference.manager import Manager
from skvaider.manifest import ManifestRequest
from skvaider.typing import JSONObject

router = APIRouter()
log = structlog.get_logger()


@router.get("/manager/health")
async def health(  # pyright: ignore[reportUnusedFunction]
    services: svcs.fastapi.DepContainer,
) -> JSONResponse:
    """
    Signal whether this server inference server is ready to be talked to.

    The state of the individual models doesn't necessarily indicate that
    this server is overall broken.

    This method is used by the proxy to determine whether this server can
    generally be talked to.

    """
    manager = services.get(Manager)
    content: JSONObject = {
        "status": "ok",
        "models": {
            m.config.id: {"status": list(sorted(m.status))}
            for m in manager.list_models()
        },
    }
    return JSONResponse(status_code=200, content=content)


@router.patch("/manager/manifest")
async def update_manifest(
    body: ManifestRequest,
    services: svcs.fastapi.DepContainer,
) -> JSONResponse:
    """Update the manifest of models to load; the manager converges asynchronously."""
    manager = services.get(Manager)
    manager.update_manifest(body.models, body.serial)
    return JSONResponse(status_code=202, content={"status": "ok"})


@router.get("/manager/usage")
async def usage(
    services: svcs.fastapi.DepContainer,
) -> JSONResponse:
    """Return memory usage statistics per backend."""
    manager = services.get(Manager)
    content: JSONObject = {
        "memory": {
            monitor.id: {
                "total": monitor.total,
                "used": monitor.used,
                "free": monitor.free,
            }
            for monitor in manager.monitors.values()
        }
    }
    return JSONResponse(status_code=200, content=content)
