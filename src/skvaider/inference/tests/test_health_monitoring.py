import asyncio
import contextlib
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, Mock

import openai
import pytest

from skvaider.conftest import wait_for_condition
from skvaider.inference.config import ModelConfig
from skvaider.inference.conftest import OpenAIServerMock
from skvaider.inference.model import Model
from skvaider.utils import HealthCheckResult


@wait_for_condition()
async def has_health_status(model: Model, expected: str) -> bool:
    return model.health.status == expected


async def test_monitor_health_updates_model_status_completion(
    openai_server: OpenAIServerMock,
):
    model = Model(
        ModelConfig(
            id="test",
            max_requests=10,
            port=1000,
            task="chat",
        ),
        lambda: None,
    )
    model.health_check_interval = 0.01
    model.health_check_timeout = 0.01
    model.quick_check_interval = 0.01

    async def quick_health_ok() -> HealthCheckResult:
        return HealthCheckResult.passed()

    model._check_quick_health = quick_health_ok

    async def health_ok() -> None:
        model.health.completion = HealthCheckResult.passed()

    async def health_not_ok() -> None:
        model.health.completion = HealthCheckResult.failed("not ok")

    # Phase 1: startup, no process, no health info
    model._check_health = health_ok
    asyncio.create_task(model._monitor_health())

    # Wait for the loop to run more than once.
    @wait_for_condition()
    async def loop_ran_more_than_once() -> bool:
        return model._health_checks > 1

    await loop_ran_more_than_once()

    # The model isn't starting yet. That means we don't set a health status
    assert model.health.status == ""
    assert "inactive" in model.status

    # Phase 2: expose an endpoint so that the health check starts running
    model.process_status = "running"
    model.endpoint = openai_server.endpoint
    await has_health_status(model, "healthy")
    assert model.status == {"running", "healthy", "active"}

    # Phase 3: model fails, becomes unhealthy and inactive
    model._check_health = health_not_ok
    await has_health_status(model, "unhealthy")
    assert model.status == {"running", "unhealthy", "inactive"}

    # Phase 4: model recovers, becomes healthy and active again
    model._check_health = health_ok
    await has_health_status(model, "healthy")
    assert model.status == {"running", "healthy", "active"}

    # Phase 5: model check fails with an exception
    async def health_exception() -> None:
        raise Exception()

    model._check_health = health_exception
    await has_health_status(model, "unhealthy")
    assert model.status == {"running", "unhealthy", "inactive"}

    # Phase 6: recover again
    model._check_health = health_ok
    await has_health_status(model, "healthy")
    assert model.status == {"running", "healthy", "active"}


@pytest.fixture
def fast_model() -> Model:
    """A model with fast check intervals and healthy check mocks."""
    model = Model(
        ModelConfig(id="test", max_requests=10, port=1000, task="chat"),
        lambda: None,
    )
    model.quick_check_interval = 0.01
    model.health_check_timeout = 0.01
    model._check_quick_health = AsyncMock(
        return_value=HealthCheckResult.passed()
    )

    async def deep_ok(health: ModelHealth) -> None:
        health.completion = HealthCheckResult.passed()

    model._check_health = deep_ok
    model.process_status = "running"
    model.endpoint = "http://"
    return model


@contextlib.asynccontextmanager
async def monitoring(model: Model) -> AsyncGenerator[None, None]:
    task = asyncio.create_task(model._monitor_health())
    try:
        yield
    finally:
        task.cancel()


async def test_monitor_health_quick_check_failure_is_unhealthy(
    fast_model: Model,
):
    async with monitoring(fast_model):
        await has_health_status(fast_model, "healthy")

        fast_model._check_quick_health = AsyncMock(
            return_value=HealthCheckResult.failed("connection refused")
        )
        await has_health_status(fast_model, "unhealthy")
        assert fast_model.health.quick_health_check.message == (
            "connection refused"
        )
        assert fast_model.health.completion.ok


async def test_monitor_health_runs_deep_check_once_per_interval(
    fast_model: Model,
):
    deep_check = AsyncMock(return_value=None)
    fast_model._check_health = deep_check
    async with monitoring(fast_model):
        await has_health_status(fast_model, "healthy")
        loops = fast_model._health_checks
        await asyncio.sleep(0.2)
        assert fast_model._health_checks > loops + 2

    deep_check.assert_awaited_once()


async def test_monitor_health_keeps_deep_check_result_between_runs(
    fast_model: Model,
):
    async def deep_fails(health: ModelHealth) -> None:
        health.completion = HealthCheckResult.failed("HTTP 500")

    deep_check = AsyncMock(side_effect=deep_fails)
    fast_model._check_health = deep_check
    async with monitoring(fast_model):
        await has_health_status(fast_model, "unhealthy")
        # Quick checks keep passing, but the failed deep check must keep the
        # model unhealthy until the next deep check.
        await asyncio.sleep(0.2)
        assert fast_model.health.status == "unhealthy"
        assert fast_model.health.completion.message == "HTTP 500"

    deep_check.assert_awaited_once()


async def test_monitor_health_does_not_increase_deep_check_frequency_after_exception(
    fast_model: Model,
):
    deep_check = AsyncMock(side_effect=[Exception("boom"), None])
    fast_model._check_health = deep_check
    async with monitoring(fast_model):
        await asyncio.sleep(1)
    assert deep_check.await_count == 1
    assert fast_model._health_checks > 4
    # The failure marker is kept
    assert fast_model.health.deep_check_failure.message == "boom"
    assert fast_model.health.quick_health_check.ok


async def test_monitor_health_resets_status_without_endpoint(
    fast_model: Model,
):
    async with monitoring(fast_model):
        await has_health_status(fast_model, "healthy")
        fast_model.endpoint = None
        await has_health_status(fast_model, "")


async def test_check_quick_health(openai_server: OpenAIServerMock):
    model = Model(
        ModelConfig(
            id="test", port=openai_server.port, max_requests=10, task="chat"
        ),
        lambda: None,
    )
    model.endpoint = openai_server.endpoint
    result = await model._check_quick_health()
    assert result.ok
    assert result.message == ""

    openai_server.response_status = 503
    result = await model._check_quick_health()
    assert not result.ok
    assert result.message.startswith("503: ")

    model.endpoint = "http://127.0.0.1:1"
    result = await model._check_quick_health()
    assert not result.ok
    assert result.message


async def test_health_check_embeddings(openai_server: OpenAIServerMock):
    model = Model(
        ModelConfig(
            id="test-embed",
            port=openai_server.port,
            max_requests=10,
            task="embedding",
        ),
        lambda: None,
    )
    model.endpoint = openai_server.endpoint

    expected_embedding = [0.1, 0.2, 0.3]
    openai_server.response = {
        "model": "test-embed",
        "object": "list",
        "data": [
            {"embedding": expected_embedding, "index": 0, "object": "embedding"}
        ],
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }

    # 1. No verification data -> embedding ok, numerical warns
    await model._check_embedding_health()
    assert model.health.embedding is not None and model.health.embedding.ok
    assert model.health.numerical is not None
    assert model.health.numerical.failure_message == "no reference data"

    # 2. With verification data, correct embedding -> all ok
    model.verification_data = {"test input": expected_embedding}
    await model._check_embedding_health()
    assert not model.health.has_failures()
    assert openai_server.last_request_json["input"] == ["test input"]

    # 3. Wrong embedding -> numerical fails
    openai_server.response["data"][0]["embedding"] = [0.9, 0.9, 0.9]
    await model._check_embedding_health()
    assert model.health.has_failures()


async def test_health_check_completions(openai_server: OpenAIServerMock):
    model = Model(
        ModelConfig(
            id="test-chat",
            port=openai_server.port,
            max_requests=10,
            task="chat",
        ),
        lambda: None,
    )
    model.endpoint = openai_server.endpoint

    # 2. Simulate correct response -> healthy
    openai_server.response = {}
    assert (await model._check_completion_health()).ok
    assert openai_server.last_request_json["prompt"] == "2+2="

    # 2. Simulate wrong response -> unhealthy
    openai_server.response_status = 500
    assert not (await model._check_completion_health()).ok
    assert openai_server.last_request_json["prompt"] == "2+2="


async def test_health_check_streaming_tool_call_check_live(gemma: Model):
    await gemma.start()
    assert (await gemma._check_completion_streaming_tool_call_health()).ok


async def test_health_check_streaming_tool_call_check_offline():
    model = Model(
        ModelConfig(
            id="test-chat",
            port=9000,
            max_requests=10,
            task="chat",
        ),
        lambda: None,
    )
    model.endpoint = "http://localhost:9000"
    check = await model._check_completion_streaming_tool_call_health()
    assert check.failure_message == "Error connecting: Connection error."


async def test_health_check_streaming_tool_call_check_mocked(
    monkeypatch: pytest.MonkeyPatch,
):
    model = Model(
        ModelConfig(
            id="test-chat",
            port=8100,
            max_requests=10,
            task="chat",
        ),
        lambda: None,
    )
    model.endpoint = "http://localhost:9999"

    # mock openai
    openai_mock = Mock()
    openai_mock.return_value = client = AsyncMock()
    monkeypatch.setattr(openai, "AsyncOpenAI", openai_mock)

    class AsyncCmMock:
        def __init__(self, mock: Any):
            self.mock = mock

        async def __aenter__(self):
            return self.mock

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any):
            pass

    stream = AsyncMock()
    client.chat.completions.stream = Mock(return_value=AsyncCmMock(stream))
    stream.get_final_completion.return_value = final = Mock()
    stream.__aiter__.return_value = [1, 2, 3]
    choice = Mock()
    choice.message.tool_calls = []
    final.choices = [choice]
    # no tool calls returned
    check = await model._check_completion_streaming_tool_call_health()
    assert check.failure_message == "No tool calls found"

    # tool call without id returned
    call = Mock()
    call.id = None
    choice.message.tool_calls = [call]
    check = await model._check_completion_streaming_tool_call_health()
    assert check.failure_message == "Missing tool call ID in tool call #0: None"

    # no function name
    call.id = "1234"
    call.function.name = None
    check = await model._check_completion_streaming_tool_call_health()
    assert (
        check.failure_message
        == "Missing tool function name in tool call #0: None"
    )

    # incorrect json args
    call.id = "1234"
    call.function.name = "Asdf"
    call.function.arguments = r'{"object": 123'
    check = await model._check_completion_streaming_tool_call_health()
    assert (
        check.failure_message
        == "Invalid argument JSON in tool call #0: '{\"object\": 123'"
    )


def make_mock_process(returncode: int = -9) -> Mock:
    process = Mock()
    process.wait = AsyncMock(return_value=None)
    process.returncode = returncode
    process.pid = 12345
    return process


async def test_monitor_process_calls_on_unexpected_exit_on_crash():
    called: list[bool] = []
    model = Model(
        ModelConfig(id="test", max_requests=10, port=1000, task="chat"),
        lambda: called.append(True),
    )
    model.process = make_mock_process()
    model.process_status = "running"

    await model._monitor_process()

    assert called == [True]
    assert model.process_status == "stopped"


async def test_monitor_process_skips_on_unexpected_exit_on_normal_stop():
    called: list[bool] = []
    model = Model(
        ModelConfig(id="test", max_requests=10, port=1000, task="chat"),
        lambda: called.append(True),
    )
    model.process = make_mock_process(returncode=0)
    model.process_status = "stopping"

    await model._monitor_process()

    assert called == []


async def test_monitor_process_skips_on_unexpected_exit_on_abnormal_stop():
    # A non-zero exit code while stopping (e.g. SIGTERM) is still expected —
    # the callback must not fire.
    called: list[bool] = []
    model = Model(
        ModelConfig(id="test", max_requests=10, port=1000, task="chat"),
        lambda: called.append(True),
    )
    model.process = make_mock_process(returncode=-15)
    model.process_status = "stopping"

    await model._monitor_process()

    assert called == []
