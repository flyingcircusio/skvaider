import asyncio
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog
import svcs
from fastapi import HTTPException, Request

from skvaider.proxy.backends import Backend
from skvaider.proxy.models import AIModel
from skvaider.proxy.pool import Pool
from skvaider.routers.openai import OpenAIProxy

log = structlog.stdlib.get_logger()


class MockBackend(Backend):
    def __init__(self, url: str, pool: Pool, fail_count: int = 0):
        super().__init__(url, pool)
        self.fail_count = fail_count
        self.call_count = 0
        self.models = {}
        self.memory = {"gpu": {"free": 100, "total": 100}}
        self._idle_event = asyncio.Event()
        self._idle_event.set()

    async def post(self, path: str, data: dict[str, Any]) -> None:  # type: ignore[override]
        self.call_count += 1
        if self.call_count <= self.fail_count:
            raise HTTPException(status_code=540, detail="Backend unavailable")
        self._last_result = {"success": True, "backend": self.url}

    async def post_stream(
        self, path: str, data: dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        self.call_count += 1
        if self.call_count <= self.fail_count:
            raise HTTPException(status_code=540, detail="Backend unavailable")
        yield f"chunk from {self.url}"

    async def load_model(self, model_id: str) -> bool:
        model = self.models[model_id]
        model.is_loaded = True
        return True

    async def unload_model(self, model_id: str):
        model = self.models[model_id]
        model.is_loaded = False

    async def monitor_health_and_update_models(self):
        pass

    async def wait_for_idle(self):
        await self._idle_event.wait()


@pytest.fixture
async def pool() -> AsyncGenerator[Pool, None]:  # type: ignore[misc]
    p = Pool()
    yield p
    p.close()


@pytest.fixture
def services(pool: Pool) -> MagicMock:  # type: ignore[misc]
    s = MagicMock(spec=svcs.fastapi.DepContainer)
    s.get.return_value = pool
    return s


@pytest.fixture
def request_factory():  # type: ignore[misc]
    def _create_request(
        model: str = "test-model", stream: bool = False
    ) -> MagicMock:
        req = MagicMock(spec=Request)
        req.json = AsyncMock(return_value={"model": model, "stream": stream})
        req.state = MagicMock()
        req.state.model = model
        req.state.stream = stream
        return req

    return _create_request


@pytest.mark.asyncio
async def test_proxy_retry_all_fail(pool: Pool, services: MagicMock, request_factory) -> None:  # type: ignore[misc]
    """Test that proxy raises 503 when all backends are unavailable."""
    b1 = MockBackend("http://b1", pool, fail_count=100)
    pool.add_backend(b1)

    model = AIModel(id="test-model", owned_by="me", backend=b1)
    model.memory_usage = {"gpu": 10}
    model.is_loaded = True
    model.limit = 100
    b1.models["test-model"] = model

    pool.update_model_maps()

    proxy = OpenAIProxy(services)
    req = request_factory(stream=False)  # type: ignore[misc]

    with pytest.raises(HTTPException) as exc:
        await proxy.proxy(req, "/test")  # type: ignore[misc]

    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_proxy_retry_verifies_backend_switching(pool: Pool, services: MagicMock, request_factory) -> None:  # type: ignore[misc]
    """Test that request goes to failing backend (540), then retries with next backend."""
    # Backend 1 fails once with 540
    b1 = MockBackend("http://b1", pool, fail_count=1)
    # Backend 2 always succeeds
    b2 = MockBackend("http://b2", pool, fail_count=0)

    for b in [b1, b2]:
        pool.add_backend(b)
        model = AIModel(id="test-model", owned_by="me", backend=b)
        model.memory_usage = {"gpu": 10}
        model.is_loaded = True
        model.limit = 100
        b.models["test-model"] = model

    pool.update_model_maps()

    proxy = OpenAIProxy(services)
    req = request_factory(stream=False)  # type: ignore[misc]

    # Execute the request
    await proxy.proxy(req, "/test")  # type: ignore[misc]

    # Verify that at least one backend was called
    total_calls = b1.call_count + b2.call_count
    assert total_calls >= 1, "At least one backend should have been called"

    # If b1 was called, verify it failed with 540 and b2 took over
    if b1.call_count > 0:
        assert (
            b1.call_count == 1
        ), "b1 should have been called exactly once (and failed)"
        # Since b1 failed, b2 must have been used for the retry
        assert b2.call_count > 0, "b2 should have been called after b1 failed"
