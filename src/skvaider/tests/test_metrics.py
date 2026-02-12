"""Tests for gateway Prometheus metrics instrumentation."""

import asyncio
from typing import Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
import svcs
from fastapi import HTTPException, Request
from prometheus_client import REGISTRY

from skvaider.config import ModelInstanceConfig
from skvaider.proxy.backends import Backend
from skvaider.proxy.models import AIModel
from skvaider.proxy.pool import Pool
from skvaider.routers.openai import OpenAIProxy


def prometheus_value(metric: str, **labels: str) -> float | None:
    return REGISTRY.get_sample_value(metric, labels)


class MockBackend(Backend):
    """Minimal mock backend for metrics tests."""

    def __init__(self, url: str, fail_count: int = 0):
        super().__init__(url)
        self.fail_count = fail_count
        self.call_count = 0
        self.models = {}
        self.healthy = True
        self.memory = {"gpu": {"free": 100, "total": 100}}
        self._idle_event = asyncio.Event()
        self._idle_event.set()

    async def post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:  # type: ignore[override]
        self.call_count += 1
        if self.call_count <= self.fail_count:
            raise HTTPException(status_code=540, detail="Backend unavailable")
        return {"id": "cmpl-1", "choices": []}

    async def post_stream(
        self, path: str, data: dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        self.call_count += 1
        if self.call_count <= self.fail_count:
            raise HTTPException(status_code=540, detail="Backend unavailable")
        yield f"data: chunk from {self.url}\n\n"

    async def load_model(self, model_id: str) -> bool:
        model = self.models[model_id]
        model.is_loaded = True
        return True

    async def unload_model(self, model_id: str) -> None:
        model = self.models[model_id]
        model.is_loaded = False

    async def monitor_health_and_update_models(self) -> None:
        pass


@pytest.fixture
def simple_request() -> MagicMock:
    req = MagicMock(spec=Request)
    req.json = AsyncMock(return_value={"model": "test-model", "stream": False})
    req.state = MagicMock()
    req.state.model = "test-model"
    req.state.stream = False
    return req


@pytest.fixture
def streaming_request() -> MagicMock:
    req = MagicMock(spec=Request)
    req.json = AsyncMock(return_value={"model": "test-model", "stream": True})
    req.state = MagicMock()
    req.state.model = "test-model"
    req.state.stream = True
    return req


@pytest.fixture
def backend():
    return MockBackend("http://backend-1")


@pytest.fixture
async def pool(
    svcs_registry: svcs.Registry, backend: MockBackend
) -> AsyncGenerator[Pool, None]:
    pool = Pool(
        model_configs=[
            ModelInstanceConfig(
                id="test-model", instances=1, memory={"gpu": 10}
            )
        ],
        backends=[backend],
    )

    for b in pool.backends:
        model = AIModel(id="test-model", owned_by="test", backend=b)
        model.memory_usage = {"gpu": 10}
        model.is_loaded = True
        b.models["test-model"] = model

    await pool.rebalance()

    svcs_registry.register_value(  # pyright: ignore[reportUnknownMemberType]
        Pool, pool
    )

    yield pool

    pool.close()


@pytest.fixture
def proxy(
    pool: Pool, services: svcs.Container
) -> Generator[OpenAIProxy, None, None]:
    yield OpenAIProxy(services)


async def test_requests_total_increments_on_success(
    proxy: OpenAIProxy, simple_request: MagicMock
):
    await proxy.proxy(simple_request, "/v1/chat/completions")

    value = prometheus_value(
        "skvaider_gateway_requests_total",
        model="test-model",
        endpoint="/v1/chat/completions",
        streaming="False",
        status="success",
    )
    assert value == 1


async def test_requests_total_increments_on_error(
    pool: Pool,
    proxy: OpenAIProxy,
    backend: MockBackend,
    simple_request: MagicMock,
):
    """All backends fail with 540 → 503 raised → status='error'."""
    backend.fail_count = 100

    with pytest.raises(HTTPException) as exc:
        await proxy.proxy(simple_request, "/v1/chat/completions")
    assert exc.value.status_code == 503

    value = prometheus_value(
        "skvaider_gateway_requests_total",
        model="test-model",
        streaming="False",
        endpoint="/v1/chat/completions",
        status="error",
    )
    assert value == 1


async def test_backend_requests_total_success(
    proxy: OpenAIProxy, simple_request: MagicMock
):
    await proxy.proxy(simple_request, "/v1/chat/completions")

    value = prometheus_value(
        "skvaider_gateway_backend_requests_total",
        model="test-model",
        backend="http://backend-1",
        endpoint="/v1/chat/completions",
        status="success",
        streaming="False",
    )
    assert value == 1


async def test_backend_requests_total_error_on_failure(
    pool: Pool,
    proxy: OpenAIProxy,
    backend: MockBackend,
    simple_request: MagicMock,
):
    """All backends fail → backend error counter goes up."""
    backend.fail_count = 100

    with pytest.raises(HTTPException):
        await proxy.proxy(simple_request, "/v1/chat/completions")

    value = prometheus_value(
        "skvaider_gateway_backend_requests_total",
        model="test-model",
        backend="http://backend-1",
        endpoint="/v1/chat/completions",
        status="error",
        streaming="False",
    )
    assert value == 1


async def test_retry_total_increments_on_backend_unavailable(
    pool: Pool,
    proxy: OpenAIProxy,
    backend: MockBackend,
    simple_request: MagicMock,
):
    """Backend fails once with 540, retry happens → retry counter increments."""
    backend.fail_count = 1

    b2 = MockBackend("http://backend-2")
    b2.pool = pool
    model = AIModel(id="test-model", owned_by="test", backend=b2)
    model.memory_usage = {"gpu": 10}
    model.is_loaded = True
    b2.models["test-model"] = model
    pool.backends.append(b2)

    await proxy.proxy(simple_request, "/v1/chat/completions")

    value = prometheus_value(
        "skvaider_gateway_backend_retry_total",
        model="test-model",
        backend="http://backend-1",
        endpoint="/v1/chat/completions",
        reason="backend_unavailable",
        streaming="False",
    )
    assert value == 1


async def test_request_duration_is_observed(
    proxy: OpenAIProxy, simple_request: MagicMock
):
    await proxy.proxy(simple_request, "/v1/chat/completions")

    value = prometheus_value(
        "skvaider_gateway_request_duration_seconds_bucket",
        model="test-model",
        endpoint="/v1/chat/completions",
        streaming="False",
        le="0.5",
    )

    assert value == 1


async def test_active_requests_returns_to_zero_after_success(
    proxy: OpenAIProxy, simple_request: MagicMock
):
    await proxy.proxy(simple_request, "/v1/chat/completions")

    value = prometheus_value(
        "skvaider_gateway_active_requests",
        model="test-model",
        endpoint="/v1/chat/completions",
        streaming="False",
    )
    assert value == 0


async def test_active_requests_returns_to_zero_after_error(
    pool: Pool,
    proxy: OpenAIProxy,
    backend: MockBackend,
    simple_request: MagicMock,
):
    backend.fail_count = 100

    with pytest.raises(HTTPException):
        await proxy.proxy(simple_request, "/v1/chat/completions")

    value = prometheus_value(
        "skvaider_gateway_active_requests",
        model="test-model",
        endpoint="/v1/chat/completions",
        streaming="False",
    )
    assert value == 0


async def test_streaming_request_increments_metrics(
    proxy: OpenAIProxy, streaming_request: MagicMock
):
    """Streaming requests should also update gateway_requests_total."""
    result = await proxy.proxy(streaming_request, "/v1/chat/completions")

    async for _ in result.body_iterator:
        pass

    value = prometheus_value(
        "skvaider_gateway_requests_total",
        model="test-model",
        streaming="True",
        endpoint="/v1/chat/completions",
        status="success",
    )
    assert value == 1
