from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog
import svcs
from fastapi import HTTPException, Request

from skvaider.config import ModelInstanceConfig, parse_size
from skvaider.proxy.backends import Backend
from skvaider.proxy.models import AIModel
from skvaider.proxy.pool import Pool
from skvaider.routers.openai import OpenAIProxy
from skvaider.utils import TaskManager

log = structlog.stdlib.get_logger()


class MockBackend(Backend):
    def __init__(self, url: str, fail_count: int = 0):
        super().__init__(url)
        self.fail_count = fail_count
        self.call_count = 0
        self.models = {}
        self.memory = {
            "ram": {"free": parse_size("100K"), "total": parse_size("100K")}
        }

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
async def test_proxy_retry_all_fail(
    request_factory,  # type: ignore[misc]
    task_managers: list[TaskManager],
) -> None:
    """Test that proxy raises 503 when all backends are unavailable."""
    backend = MockBackend("http://b1", fail_count=100)
    backend.healthy = True

    model = AIModel(id="test-model", owned_by="me", backend=backend)
    model.memory_usage = {"ram": parse_size("10K")}
    model.is_loaded = True
    backend.models["test-model"] = model

    model_config = ModelInstanceConfig(
        id="test-model", instances=1, memory={"ram": parse_size("10K")}
    )
    pool = Pool([model_config], [backend])
    task_managers.append(pool.tasks)

    await pool.rebalance()

    services = MagicMock(spec=svcs.fastapi.DepContainer)
    services.get.return_value = pool

    proxy = OpenAIProxy(services)
    req = request_factory(stream=False)  # type: ignore[misc]

    with pytest.raises(HTTPException) as exc:
        await proxy.proxy(req, "/test")  # type: ignore[misc]

    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_proxy_retry_verifies_backend_switching(
    request_factory,  # type: ignore[misc]
    task_managers: list[TaskManager],
) -> None:
    """Test that request goes to failing backend (540), then retries with next backend."""
    pool = Pool(
        [
            ModelInstanceConfig(
                id="test-model", instances=2, memory={"ram": parse_size("10K")}
            )
        ]
    )
    task_managers.append(pool.tasks)

    b1 = MockBackend("http://b1", fail_count=1)
    b1.healthy = True
    b2 = MockBackend("http://b2", fail_count=0)
    b2.healthy = True

    for b in [b1, b2]:
        b.pool = pool
        pool.backends.append(b)
        pool.tasks.create(b.monitor_health_and_update_models)
        model = AIModel(id="test-model", owned_by="me", backend=b)
        model.memory_usage = {"ram": parse_size("10K")}
        model.is_loaded = True
        model.limit = 100
        b.models["test-model"] = model

    await pool.rebalance()

    services = MagicMock(spec=svcs.fastapi.DepContainer)
    services.get.return_value = pool

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
