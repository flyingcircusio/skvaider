"""Admin endpoints — accessible only via admin tokens."""

from typing import Literal

import svcs
from fastapi import APIRouter, Security
from pydantic import BaseModel

from skvaider.auth import verify_admin_token
from skvaider.proxy.pool import Pool

router = APIRouter()


class CheckResult(BaseModel):
    status: Literal["ok", "warning", "critical"]
    message: str


class HealthResponse(BaseModel):
    status: Literal["ok", "warning", "critical"]
    checks: dict[str, CheckResult]


@router.get("/health")
async def health(
    services: svcs.fastapi.DepContainer,
    _: None = Security(verify_admin_token),
) -> HealthResponse:
    pool = services.get(Pool)
    checks: dict[str, CheckResult] = {}

    # Every configured model must have at least one loaded instance somewhere.
    for model_id, model_config in pool.model_configs.items():
        loaded = pool.count_loaded_instances(model_id)
        desired = model_config.instances
        name = f"model_loaded[{model_id}]"
        if loaded == 0:
            checks[name] = CheckResult(
                status="critical",
                message=f"no instances loaded (desired {desired})",
            )
        elif loaded < desired:
            checks[name] = CheckResult(
                status="warning",
                message=f"{loaded}/{desired} instances loaded",
            )
        else:
            checks[name] = CheckResult(status="ok", message=f"{loaded} loaded")

    # At least one backend must be healthy.
    healthy_backends = [b for b in pool.backends if b.healthy]
    all_backends = pool.backends
    if not all_backends:
        checks["backends"] = CheckResult(
            status="critical", message="no backends configured"
        )
    elif not healthy_backends:
        checks["backends"] = CheckResult(
            status="critical",
            message=f"all {len(all_backends)} backend(s) unhealthy",
        )
    elif len(healthy_backends) < len(all_backends):
        checks["backends"] = CheckResult(
            status="warning",
            message=(
                f"{len(healthy_backends)}/{len(all_backends)} backends healthy"
            ),
        )
    else:
        checks["backends"] = CheckResult(
            status="ok", message=f"{len(all_backends)} backend(s) healthy"
        )

    # No active model should exceed its configured memory.
    for backend in pool.backends:
        for model in backend.models.values():
            if not model.is_loaded:
                continue
            overages = model.check_memory_usage()
            for resource, (actual, configured) in overages.items():
                name = f"memory[{model.id}@{backend.url},{resource}]"
                checks[name] = CheckResult(
                    status="warning",
                    message=(
                        f"actual {actual} > configured {configured} bytes"
                    ),
                )

    if any(c.status == "critical" for c in checks.values()):
        overall = "critical"
    elif any(c.status == "warning" for c in checks.values()):
        overall = "warning"
    else:
        overall = "ok"

    return HealthResponse(status=overall, checks=checks)
