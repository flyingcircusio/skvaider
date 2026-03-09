import pytest
from fastapi import HTTPException
from prometheus_client import REGISTRY

from skvaider.proxy.backends import DummyBackend
from skvaider.proxy.models import AIModel
from skvaider.proxy.pool import Pool
from skvaider.routers.openai import OpenAIProxy


def prometheus_value(metric: str, **labels: str) -> float | None:
    return REGISTRY.get_sample_value(metric, labels)


async def test_requests_total_increments_on_success(
    proxy: OpenAIProxy, mock_request_factory  # type: ignore[misc]
):
    await proxy.proxy(mock_request_factory(), "/v1/chat/completions")  # type: ignore[misc]

    value = prometheus_value(
        "skvaider_gateway_requests_total",
        model="test-model",
        endpoint="/v1/chat/completions",
        streaming="False",
        status="success",
    )
    assert value == 1


async def test_requests_total_increments_on_error(
    proxy: OpenAIProxy,
    dummy_backend: DummyBackend,
    mock_request_factory,  # type: ignore[misc]
):
    dummy_backend.fail_count = 100

    with pytest.raises(HTTPException) as exc:
        await proxy.proxy(mock_request_factory(), "/v1/chat/completions")  # type: ignore[misc]
    assert exc.value.status_code == 503

    value = prometheus_value(
        "skvaider_gateway_requests_total",
        model="test-model",
        streaming="False",
        endpoint="/v1/chat/completions",
        status="error",
    )
    assert value == 1


async def test_backend_requests_total_success(proxy: OpenAIProxy, mock_request_factory):  # type: ignore[misc]
    await proxy.proxy(mock_request_factory(), "/v1/chat/completions")  # type: ignore[misc]

    value = prometheus_value(
        "skvaider_gateway_backend_requests_total",
        model="test-model",
        backend="http://test-backend",
        endpoint="/v1/chat/completions",
        status="success",
        streaming="False",
    )
    assert value == 1


async def test_backend_requests_total_error_on_failure(
    proxy: OpenAIProxy,
    dummy_backend: DummyBackend,
    mock_request_factory,  # type: ignore[misc]
):
    dummy_backend.fail_count = 100

    with pytest.raises(HTTPException):
        await proxy.proxy(mock_request_factory(), "/v1/chat/completions")  # type: ignore[misc]

    value = prometheus_value(
        "skvaider_gateway_backend_requests_total",
        model="test-model",
        backend="http://test-backend",
        endpoint="/v1/chat/completions",
        status="error",
        streaming="False",
    )
    assert value == 1


async def test_retry_total_increments_on_backend_unavailable(
    pool: Pool,
    proxy: OpenAIProxy,
    dummy_backend: DummyBackend,
    mock_request_factory,  # type: ignore[misc]
):
    dummy_backend.fail_count = 1

    b2 = DummyBackend("http://test-backend-2")
    b2.pool = pool
    b2.healthy = True
    model = AIModel(id="test-model", owned_by="test", backend=b2)
    model.memory_usage = {"ram": 10}
    model.is_loaded = True
    b2.models["test-model"] = model
    pool.backends.append(b2)

    await proxy.proxy(mock_request_factory(), "/v1/chat/completions")  # type: ignore[misc]

    value = prometheus_value(
        "skvaider_gateway_backend_retry_total",
        model="test-model",
        backend="http://test-backend",
        endpoint="/v1/chat/completions",
        reason="backend_unavailable",
        streaming="False",
    )
    assert value == 1


async def test_request_duration_is_observed(proxy: OpenAIProxy, mock_request_factory):  # type: ignore[misc]
    await proxy.proxy(mock_request_factory(), "/v1/chat/completions")  # type: ignore[misc]

    value = prometheus_value(
        "skvaider_gateway_request_duration_seconds_bucket",
        model="test-model",
        endpoint="/v1/chat/completions",
        streaming="False",
        le="0.5",
    )
    assert value == 1


async def test_active_requests_returns_to_zero_after_success(
    proxy: OpenAIProxy, mock_request_factory  # type: ignore[misc]
):
    await proxy.proxy(mock_request_factory(), "/v1/chat/completions")  # type: ignore[misc]

    value = prometheus_value(
        "skvaider_gateway_active_requests",
        model="test-model",
        endpoint="/v1/chat/completions",
        streaming="False",
    )
    assert value == 0


async def test_active_requests_returns_to_zero_after_error(
    proxy: OpenAIProxy,
    dummy_backend: DummyBackend,
    mock_request_factory,  # type: ignore[misc]
):
    dummy_backend.fail_count = 100

    with pytest.raises(HTTPException):
        await proxy.proxy(mock_request_factory(), "/v1/chat/completions")  # type: ignore[misc]

    value = prometheus_value(
        "skvaider_gateway_active_requests",
        model="test-model",
        endpoint="/v1/chat/completions",
        streaming="False",
    )
    assert value == 0


async def test_streaming_request_increments_metrics(
    proxy: OpenAIProxy, mock_request_factory  # type: ignore[misc]
):
    req = mock_request_factory(stream=True)  # type: ignore[misc]
    result = await proxy.proxy(
        req, "/v1/chat/completions"  # type: ignore[misc]
    )

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
