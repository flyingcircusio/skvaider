import asyncio
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any

from skvaider.proxy.backends import ModelConfig
from skvaider.routers.openai import time_backend_chunks


def test_model_config():
    cfg = ModelConfig({})
    assert cfg.get("gemma3b") == {}

    cfg.config["gemma3b"] = {"num_ctx": 3072}
    assert cfg.get("gemma3b") == {"num_ctx": 3072}

    assert cfg.get("gemma3b:latest") == {"num_ctx": 3072}
    cfg.config["gemma3b"] = {"num_ctx": 2048}
    assert cfg.get("gemma3b:latest") == {"num_ctx": 2048}

    cfg.config["__default__"] = {"num_ctx": 1024}
    assert cfg.get("unknown") == {"num_ctx": 1024}


async def test_time_backend_chunks_counts_only_backend_waits():
    async def backend() -> AsyncGenerator[str]:
        for chunk in ["a", "b"]:
            await asyncio.sleep(0.05)
            yield chunk

    request: Any = SimpleNamespace(state=SimpleNamespace(time_server=0.0))
    chunks: list[str] = []
    async for chunk in time_backend_chunks(request, backend()):
        chunks.append(chunk)
        # Time spent downstream (proxy/client) must not count as server time.
        await asyncio.sleep(0.2)

    assert chunks == ["a", "b"]
    assert 0.1 <= request.state.time_server < 0.3
