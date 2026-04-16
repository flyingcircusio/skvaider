"""Admin endpoints — accessible only via admin tokens."""

from typing import Literal

import svcs
from fastapi import APIRouter
from pydantic import BaseModel

from skvaider.config import Config
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
) -> HealthResponse:
    pool = services.get(Pool)
    config = services.get(Config)
    checks: dict[str, CheckResult] = {}

    # Memory: warn if any active model on any backend exceeds its configured limit.
    # This is read directly from backend-reported state — no HTTP needed.
    for backend in pool.backends:
        for model in backend.models.values():
            if not model.is_loaded:
                continue
            for resource, (
                actual,
                configured,
            ) in model.check_memory_usage().items():
                name = f"memory[{model.id}@{backend.url},{resource}]"
                checks[name] = CheckResult(
                    status="warning",
                    message=f"{actual} > configured {configured} bytes",
                )

    # Functional probes: call each backend that has a loaded model directly,
    # bypassing the gateway's own routing to avoid loopback reentrancy.
    model_configs = {m.id: m for m in config.models}
    for backend in pool.backends:
        for model in backend.models.values():
            if not model.is_loaded:
                continue
            mid = model.id
            task = model_configs[mid].task if mid in model_configs else "chat"

            if task == "embedding":
                try:
                    result = await backend.post(
                        "/openai/v1/embeddings",
                        {
                            "model": mid,
                            "input": "The food was delicious and the waiter...",
                            "encoding_format": "float",
                        },
                    )
                    assert result["object"] == "list"
                    assert len(result["data"]) >= 1
                    assert result["data"][0]["object"] == "embedding"
                    assert len(result["data"][0]["embedding"]) > 64
                    assert isinstance(result["data"][0]["embedding"][0], float)
                    checks[f"check_embeddings[{mid}]"] = CheckResult(
                        status="ok", message="ok"
                    )
                except Exception as e:
                    checks[f"check_embeddings[{mid}]"] = CheckResult(
                        status="critical", message=str(e)
                    )
            else:
                try:
                    result = await backend.post(
                        "/openai/v1/chat/completions",
                        {
                            "model": mid,
                            "messages": [{"role": "user", "content": "Hello"}],
                            "stream": False,
                            "max_tokens": 1000,
                        },
                    )
                    assert result["object"] == "chat.completion"
                    msg = result["choices"][0]["message"]
                    assert "content" in msg
                    assert "role" in msg
                    checks[f"check_chat_completions[{mid}]"] = CheckResult(
                        status="ok", message="ok"
                    )
                except Exception as e:
                    checks[f"check_chat_completions[{mid}]"] = CheckResult(
                        status="critical", message=str(e)
                    )

                try:
                    result = await backend.post(
                        "/openai/v1/completions",
                        {
                            "model": mid,
                            "prompt": "say hello",
                            "stream": False,
                            "max_tokens": 1000,
                        },
                    )
                    assert result["object"] == "text_completion"
                    assert "text" in result["choices"][0]
                    checks[f"check_completions[{mid}]"] = CheckResult(
                        status="ok", message="ok"
                    )
                except Exception as e:
                    checks[f"check_completions[{mid}]"] = CheckResult(
                        status="critical", message=str(e)
                    )

    if any(c.status == "critical" for c in checks.values()):
        overall = "critical"
    elif any(c.status == "warning" for c in checks.values()):
        overall = "warning"
    else:
        overall = "ok"

    return HealthResponse(status=overall, checks=checks)
