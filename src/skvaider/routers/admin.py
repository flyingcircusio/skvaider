"""Admin endpoints — accessible only via admin tokens."""

import svcs
from fastapi import APIRouter

from skvaider.proxy.models import CheckResult
from skvaider.proxy.pool import Pool

router = APIRouter()


@router.get("/health")
async def health(
    services: svcs.fastapi.DepContainer,
) -> CheckResult:
    pool = services.get(Pool)

    root = CheckResult(
        name="Skvaider",
        status="ok",
        message="",
        children=[],
    )

    for model_id in pool.model_configs:
        loaded = any(
            model.is_loaded
            for backend in pool.backends
            for model in backend.models.values()
            if model.id == model_id
        )
        model_check = CheckResult(
            name=model_id,
            status="ok" if loaded else "critical",
            message="ok" if loaded else "no loaded instance",
        )
        for backend in pool.backends:
            if model_id not in backend.models:
                continue
            model_backend_state = backend.models[model_id]
            backend_check = CheckResult(
                name=backend.url,
                status="ok",
                message="",
                children=model_backend_state.checks,
            )
            backend_check.status = backend_check.highest_status().status
            model_check.children.append(backend_check)

        root.children.append(model_check)

    root.status = root.highest_status().status

    return root
