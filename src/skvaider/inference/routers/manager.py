import structlog
import svcs
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from skvaider.inference.manager import Manager
from skvaider.manifest import ManifestRequest
from skvaider.proxy.backends import BackendHealthResponse, BackendModelInfo

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

    response = BackendHealthResponse(
        status="ok",
        current_serial=manager.manifest_serial,
        usage={
            monitor.id: {
                "total": monitor.total,
                "used": monitor.used,
                "free": monitor.free,
            }
            for monitor in manager.monitors.values()
        },
        models=[
            BackendModelInfo(
                id=m.config.id,
                status=m.status,
                max_requests=m.config.max_requests,
                memory_usage={
                    monitor.id: monitor.model_usage(m)
                    for monitor in manager.monitors.values()
                },
                health_checks=m.health_checks,
            )
            for m in manager.list_models()
        ],
    )
    return JSONResponse(
        status_code=200, content=response.model_dump(mode="json")
    )


@router.patch("/manager/manifest")
async def update_manifest(
    body: ManifestRequest,
    services: svcs.fastapi.DepContainer,
) -> JSONResponse:
    """Update the manifest of models to load; the manager converges asynchronously."""
    manager = services.get(Manager)
    models = {m.lower() for m in body.models}
    unknown = models - manager.models.keys()
    if unknown:
        raise HTTPException(
            status_code=500,
            detail=f"Unknown models: {sorted(unknown)}",
        )
    manager.update_manifest(models, body.serial)
    return JSONResponse(status_code=202, content={"status": "ok"})
