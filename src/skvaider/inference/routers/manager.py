import structlog
import svcs
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from skvaider.inference.manager import Manager
from skvaider.manifest import (
    MaintenanceEnterRequest,
    MaintenanceStatus,
    ManifestRequest,
)
from skvaider.proxy.backends import BackendHealthResponse, BackendModelInfo

router = APIRouter()
log = structlog.get_logger()

LOCAL_HOSTS = frozenset({"127.0.0.1", "::1", "::ffff:127.0.0.1"})


def require_local(request: Request) -> None:
    """Maintenance control is for the local machine; the proxy only reads the state."""
    host = request.client.host if request.client else None
    if host not in LOCAL_HOSTS:
        raise HTTPException(
            status_code=403,
            detail="maintenance control is only allowed from localhost",
        )


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
        maintenance=manager.maintenance_status().phase,
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
    unknown = body.models - manager.models.keys()
    models = body.models - unknown
    manager.update_manifest(models, body.serial)
    if unknown:
        log.warning(
            "manifest accepted with errors: unknown models",
            unknown=sorted(unknown),
            manifest=sorted(models),
            known=sorted(manager.models.keys()),
        )
        return JSONResponse(
            status_code=400,
            content={"status": "unknown models", "unknown": sorted(unknown)},
        )
    return JSONResponse(status_code=202, content={"status": "ok"})


@router.get("/manager/maintenance")
async def maintenance_status(
    services: svcs.fastapi.DepContainer,
) -> MaintenanceStatus:
    """Report whether this server is in maintenance and how far the drain is."""
    return services.get(Manager).maintenance_status()


@router.post("/manager/maintenance", dependencies=[Depends(require_local)])
async def enter_maintenance(
    body: MaintenanceEnterRequest,
    services: svcs.fastapi.DepContainer,
) -> MaintenanceStatus:
    """Ask the proxy to move our models away; safe to stop once the phase is ``drained``."""
    return services.get(Manager).request_maintenance(body.reason)


@router.delete("/manager/maintenance", dependencies=[Depends(require_local)])
async def leave_maintenance(
    services: svcs.fastapi.DepContainer,
) -> MaintenanceStatus:
    """Withdraw the maintenance request; the proxy may place models here again."""
    return services.get(Manager).leave_maintenance()
