"""Tests for gateway Prometheus metrics instrumentation."""

import asyncio
import contextlib
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, Request
from prometheus_client import Counter, Gauge, Histogram

from skvaider import metrics
from skvaider.proxy.backends import Backend
from skvaider.proxy.models import AIModel
from skvaider.proxy.pool import Pool
from skvaider.routers.openai import OpenAIProxy


def _counter_value(counter: Counter, **labels: str) -> float:
    """Read the current value of a prometheus Counter with the given labels."""
    return counter.labels(
        **labels
    )._value.get()  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType, reportReturnType]


def _gauge_value(gauge: Gauge, **labels: str) -> float:
    """Read the current value of a prometheus Gauge with the given labels."""
    return gauge.labels(
        **labels
    )._value.get()  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType, reportReturnType]


def _histogram_count(histogram: Histogram, **labels: str) -> float:
    """Read the observation count of a prometheus Histogram."""
    for metric in histogram.collect():
        for sample in metric.samples:
            if sample.name.endswith("_count") and all(
                sample.labels.get(k) == v for k, v in labels.items()
            ):
                return sample.value  # pyright: ignore[reportReturnType]
    return 0


class MockBackend(Backend):
    """Minimal mock backend for metrics tests."""

    def __init__(self, url: str, pool: Pool, fail_count: int = 0):
        super().__init__(url, pool)
        self.fail_count = fail_count
        self.call_count = 0
        self.models = {}
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

    async def wait_for_idle(self) -> None:
        await self._idle_event.wait()


def _setup_pool_with_backends(
    pool: Pool, *backends: MockBackend, model_id: str = "test-model"
) -> None:
    """Register backends + a loaded model in the pool."""
    for b in backends:
        pool.add_backend(b)
        model = AIModel(id=model_id, owned_by="test", backend=b)
        model.memory_usage = {"gpu": 10}
        model.is_loaded = True
        model.limit = 100
        b.models[model_id] = model
    pool.update_model_maps()


def _make_services(pool: Pool) -> MagicMock:
    s = MagicMock(spec=["get"])
    s.get.return_value = pool
    return s


def _make_request(model: str = "test-model", stream: bool = False) -> MagicMock:
    req = MagicMock(spec=Request)
    req.json = AsyncMock(return_value={"model": model, "stream": stream})
    req.state = MagicMock()
    req.state.model = model
    req.state.stream = stream
    return req


@pytest.fixture
async def pool() -> AsyncGenerator[Pool, None]:  # type: ignore[misc]
    p = Pool()
    yield p
    p.close()


async def test_requests_total_increments_on_success(pool: Pool) -> None:
    b = MockBackend("http://metrics-b1", pool)
    _setup_pool_with_backends(pool, b)

    before = _counter_value(
        metrics.gateway_requests_total,
        model="test-model",
        endpoint="/v1/chat/completions",
        status="success",
    )

    proxy = OpenAIProxy(_make_services(pool))
    req = _make_request()
    await proxy.proxy(req, "/v1/chat/completions")

    after = _counter_value(
        metrics.gateway_requests_total,
        model="test-model",
        endpoint="/v1/chat/completions",
        status="success",
    )
    assert after == before + 1


async def test_requests_total_increments_on_error(pool: Pool) -> None:
    """All backends fail with 540 → 503 raised → status='error'."""
    b = MockBackend("http://metrics-b2", pool, fail_count=100)
    _setup_pool_with_backends(pool, b)

    before = _counter_value(
        metrics.gateway_requests_total,
        model="test-model",
        endpoint="/v1/chat/completions",
        status="error",
    )

    proxy = OpenAIProxy(_make_services(pool))
    req = _make_request()

    with pytest.raises(HTTPException) as exc:
        await proxy.proxy(req, "/v1/chat/completions")
    assert exc.value.status_code == 503

    after = _counter_value(
        metrics.gateway_requests_total,
        model="test-model",
        endpoint="/v1/chat/completions",
        status="error",
    )
    assert after == before + 1


async def test_backend_requests_total_success(pool: Pool) -> None:
    b = MockBackend("http://metrics-b3", pool)
    _setup_pool_with_backends(pool, b)

    before = _counter_value(
        metrics.gateway_backend_requests_total,
        backend="http://metrics-b3",
        status="success",
    )

    proxy = OpenAIProxy(_make_services(pool))
    req = _make_request()
    await proxy.proxy(req, "/v1/chat/completions")

    after = _counter_value(
        metrics.gateway_backend_requests_total,
        backend="http://metrics-b3",
        status="success",
    )
    assert after == before + 1


async def test_backend_requests_total_error_on_failure(pool: Pool) -> None:
    """All backends fail → backend error counter goes up."""
    b = MockBackend("http://metrics-b4", pool, fail_count=100)
    _setup_pool_with_backends(pool, b)

    before = _counter_value(
        metrics.gateway_backend_requests_total,
        backend="http://metrics-b4",
        status="error",
    )

    proxy = OpenAIProxy(_make_services(pool))
    req = _make_request()

    with pytest.raises(HTTPException):
        await proxy.proxy(req, "/v1/chat/completions")

    after = _counter_value(
        metrics.gateway_backend_requests_total,
        backend="http://metrics-b4",
        status="error",
    )
    assert after > before


async def test_retry_total_increments_on_backend_unavailable(
    pool: Pool,
) -> None:
    """Backend fails once with 540, retry happens → retry counter increments."""
    b_fail = MockBackend("http://metrics-b5", pool, fail_count=1)
    b_ok = MockBackend("http://metrics-b6", pool)
    _setup_pool_with_backends(pool, b_fail, b_ok, model_id="retry-model")

    before = _counter_value(
        metrics.gateway_backend_retry_total,
        model="retry-model",
        reason="backend_unavailable",
    )

    # Patch pool.use to guarantee the failing backend is returned first.
    original_use = pool.use
    call_count = 0

    @contextlib.asynccontextmanager
    async def ordered_use(
        model_id: str, excluded_backends: set[str] | None = None
    ) -> AsyncGenerator[Backend, None]:
        nonlocal call_count
        excluded = excluded_backends or set()
        # On first call, return failing backend; on retry, return good one.
        backends = [b_fail, b_ok]
        for b in backends:
            if b.url not in excluded:
                call_count += 1
                yield b
                return
        raise HTTPException(503, "All excluded")

    pool.use = ordered_use  # type: ignore[assignment]
    try:
        proxy = OpenAIProxy(_make_services(pool))
        req = _make_request(model="retry-model")
        await proxy.proxy(req, "/v1/chat/completions")

        after = _counter_value(
            metrics.gateway_backend_retry_total,
            model="retry-model",
            reason="backend_unavailable",
        )
        assert after >= before + 1
    finally:
        pool.use = original_use  # type: ignore[assignment]


async def test_request_duration_is_observed(pool: Pool) -> None:
    b = MockBackend("http://metrics-b7", pool)
    _setup_pool_with_backends(pool, b)

    before_count = _histogram_count(
        metrics.gateway_request_duration_seconds,
        model="test-model",
        endpoint="/v1/chat/completions",
    )

    proxy = OpenAIProxy(_make_services(pool))
    req = _make_request()
    await proxy.proxy(req, "/v1/chat/completions")

    after_count = _histogram_count(
        metrics.gateway_request_duration_seconds,
        model="test-model",
        endpoint="/v1/chat/completions",
    )

    assert after_count == before_count + 1


async def test_active_requests_returns_to_zero_after_success(
    pool: Pool,
) -> None:
    b = MockBackend("http://metrics-b8", pool)
    _setup_pool_with_backends(pool, b)

    proxy = OpenAIProxy(_make_services(pool))
    req = _make_request()
    await proxy.proxy(req, "/v1/chat/completions")

    # After a completed request, the gauge should be back to its
    # pre-request value (or zero if no other concurrent requests).
    val = _gauge_value(metrics.gateway_active_requests, model="test-model")
    assert val == 0


async def test_active_requests_returns_to_zero_after_error(
    pool: Pool,
) -> None:
    b = MockBackend("http://metrics-b9", pool, fail_count=100)
    _setup_pool_with_backends(pool, b)

    proxy = OpenAIProxy(_make_services(pool))
    req = _make_request()

    with pytest.raises(HTTPException):
        await proxy.proxy(req, "/v1/chat/completions")

    val = _gauge_value(metrics.gateway_active_requests, model="test-model")
    assert val == 0


async def test_streaming_request_increments_metrics(pool: Pool) -> None:
    """Streaming requests should also update gateway_requests_total."""
    b = MockBackend("http://metrics-b10", pool)
    _setup_pool_with_backends(pool, b)

    before = _counter_value(
        metrics.gateway_requests_total,
        model="test-model",
        endpoint="/v1/chat/completions",
        status="success",
    )

    proxy = OpenAIProxy(_make_services(pool))
    req = _make_request(stream=True)
    result = await proxy.proxy(req, "/v1/chat/completions")

    # Consume the stream to trigger the finally block
    async for _ in result.body_iterator:
        pass

    after = _counter_value(
        metrics.gateway_requests_total,
        model="test-model",
        endpoint="/v1/chat/completions",
        status="success",
    )
    assert after == before + 1
