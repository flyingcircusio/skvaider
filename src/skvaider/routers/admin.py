"""Admin endpoints — accessible only via admin tokens."""

from typing import Literal

import svcs
from fastapi import APIRouter
from pydantic import BaseModel

from skvaider.proxy.backends import Backend
from skvaider.proxy.pool import Pool

router = APIRouter()


class CheckResult(BaseModel):
    status: Literal["ok", "warning", "critical"]
    message: str


class HealthResponse(BaseModel):
    status: Literal["ok", "warning", "critical"]
    checks: dict[str, CheckResult]


async def check_embedding(backend: Backend, model_id: str) -> CheckResult:
    try:
        result = await backend.post(
            "/openai/v1/embeddings",
            {
                "model": model_id,
                "input": "The food was delicious and the waiter...",
                "encoding_format": "float",
            },
        )
        assert result["object"] == "list"
        assert len(result["data"]) >= 1
        assert result["data"][0]["object"] == "embedding"
        assert len(result["data"][0]["embedding"]) > 64
        assert isinstance(result["data"][0]["embedding"][0], float)
        return CheckResult(status="ok", message="ok")
    except Exception as e:
        return CheckResult(status="critical", message=str(e))


async def check_chat_completions(
    backend: Backend, model_id: str
) -> CheckResult:
    try:
        result = await backend.post(
            "/openai/v1/chat/completions",
            {
                "model": model_id,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
                "max_tokens": 1000,
            },
        )
        assert result["object"] == "chat.completion"
        msg = result["choices"][0]["message"]
        assert "content" in msg
        assert "role" in msg
        return CheckResult(status="ok", message="ok")
    except Exception as e:
        return CheckResult(status="critical", message=str(e))


async def check_completions(backend: Backend, model_id: str) -> CheckResult:
    try:
        result = await backend.post(
            "/openai/v1/completions",
            {
                "model": model_id,
                "prompt": "say hello",
                "stream": False,
                "max_tokens": 1000,
            },
        )
        assert result["object"] == "text_completion"
        assert "text" in result["choices"][0]
        return CheckResult(status="ok", message="ok")
    except Exception as e:
        return CheckResult(status="critical", message=str(e))


@router.get("/health")
async def health(
    services: svcs.fastapi.DepContainer,
) -> HealthResponse:
    pool = services.get(Pool)
    checks: dict[str, CheckResult] = {}

    # Memory and load state: one check per model per backend.
    for backend in pool.backends:
        for model in backend.models.values():
            name = f"model[{model.id}@{backend.url}]"
            if not model.is_loaded:
                checks[name] = CheckResult(
                    status="critical", message="not loaded"
                )
            elif model.check_memory_usage():
                checks[name] = CheckResult(
                    status="warning", message="exceeds configured memory"
                )
            else:
                checks[name] = CheckResult(status="ok", message="ok")

    # Functional probes: call each backend that has a loaded model directly,
    # bypassing the gateway's own routing to avoid loopback reentrancy.
    for backend in pool.backends:
        for model in backend.models.values():
            if not model.is_loaded:
                continue
            mid = model.id
            task = model.config.task

            if task == "embedding":
                checks[f"check_embeddings[{mid}]"] = await check_embedding(
                    backend, mid
                )
            else:
                checks[
                    f"check_chat_completions[{mid}]"
                ] = await check_chat_completions(backend, mid)
                checks[f"check_completions[{mid}]"] = await check_completions(
                    backend, mid
                )

    if any(c.status == "critical" for c in checks.values()):
        overall = "critical"
    elif any(c.status == "warning" for c in checks.values()):
        overall = "warning"
    else:
        overall = "ok"

    return HealthResponse(status=overall, checks=checks)
