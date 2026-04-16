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
            for check_name, result in model.checks.items():
                checks[f"{check_name}[{model.id}@{backend.url}]"] = result

    # Pool-level: each configured model must have at least one loaded instance.
    for model_id in pool.model_configs:
        loaded = any(
            model.is_loaded
            for backend in pool.backends
            for model in backend.models.values()
            if model.id == model_id
        )
        checks[f"loaded[{model_id}]"] = CheckResult(
            status="ok" if loaded else "critical",
            message="ok" if loaded else "no loaded instance",
        )

    if any(c.status == "critical" for c in checks.values()):
        overall = "critical"
    elif any(c.status == "warning" for c in checks.values()):
        overall = "warning"
    else:
        overall = "ok"

    return HealthResponse(status=overall, checks=checks)
