"""Admin endpoints — accessible only via admin tokens."""

from typing import Literal

import svcs
from fastapi import APIRouter
from pydantic import BaseModel

from skvaider.proxy.models import CheckResult
from skvaider.proxy.pool import Pool

router = APIRouter()


class HealthResponse(BaseModel):
    status: Literal["ok", "warning", "critical"]
    checks: dict[str, CheckResult]


@router.get("/health")
async def health(
    services: svcs.fastapi.DepContainer,
) -> HealthResponse:
    pool = services.get(Pool)
    checks: dict[str, CheckResult] = {}

    for backend in pool.backends:
        for model in backend.models.values():
            name = f"model[{model.id}@{backend.url}]"
            if not model.is_loaded:
                checks[name] = CheckResult(
                    status="critical", message="not loaded"
                )
            elif exceeding := model.check_memory_usage():
                over = ", ".join(
                    f"{r}: {actual} > {configured}"
                    for r, (actual, configured) in exceeding.items()
                )
                checks[name] = CheckResult(
                    status="warning",
                    message=f"exceeds configured memory: {over}",
                )
            else:
                checks[name] = CheckResult(status="ok", message="ok")

            if model.is_loaded and model.functional_check is not None:
                checks[f"functional[{model.id}@{backend.url}]"] = (
                    model.functional_check
                )

    if any(c.status == "critical" for c in checks.values()):
        overall = "critical"
    elif any(c.status == "warning" for c in checks.values()):
        overall = "warning"
    else:
        overall = "ok"

    return HealthResponse(status=overall, checks=checks)
