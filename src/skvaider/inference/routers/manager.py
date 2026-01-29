import structlog
import svcs
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from skvaider.inference.manager import Manager
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


@router.get("/manager/usage")
async def usage(
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
        "vram": {
            "total": manager.vram_total,
            "used": manager.vram_used,
            "free": manager.vram_free,
        }
    }
    return JSONResponse(status_code=200, content=content)
