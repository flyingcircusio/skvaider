import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock

import openai
import pytest

from skvaider.conftest import wait_for_condition
from skvaider.inference.config import ModelConfig
from skvaider.inference.conftest import OpenAIServerMock
from skvaider.inference.model import Model


@wait_for_condition()
async def has_health_status(model: Model, expected: str) -> bool:
    return model.health_status == expected


async def test_monitor_health_updates_model_status_completion():
    model = Model(
        ModelConfig(
            id="test",
            max_requests=10,
            port=1000,
            task="chat",
        )
    )
    model.health_check_interval = 0.01
    model.health_check_timeout = 0.01

    async def health_ok() -> dict[str, str]:
        return {}

    async def health_not_ok() -> dict[str, str]:
        return {"health": "not ok"}

    # Phase 1: startup, no process, no health info
    model._check_health = health_ok
    asyncio.create_task(model._monitor_health())

    await asyncio.sleep(1)  # let some time pass so the checks can actually run

    assert (
        model._health_checks > 1
    )  # ensure the loop has been run more than once

    # The model isn't starting yet. That means we don't set a health status
    assert model.health_status == ""
    assert "inactive" in model.status

    # Phase 2: expose an endpoint so that the health check starts running
    model.process_status = "running"
    model.endpoint = "http://"  # expose a dummy endpoint to trigger the check
    await has_health_status(model, "healthy")
    assert model.status == set(["running", "healthy", "active"])

    # Phase 3: model fails, becomes unhealthy and inactive
    model._check_health = health_not_ok
    await has_health_status(model, "unhealthy")
    assert model.status == set(["running", "unhealthy", "inactive"])

    # Phase 4: model recovers, becomes healthy and active again
    model._check_health = health_ok
    await has_health_status(model, "healthy")
    assert model.status == set(["running", "healthy", "active"])

    # Phase 5: model check fails with an exception
    async def health_exception():
        raise Exception()

    model._check_health = health_exception
    await has_health_status(model, "unhealthy")
    assert model.status == set(["running", "unhealthy", "inactive"])

    # Phase 6: recover again
    model._check_health = health_ok
    await has_health_status(model, "healthy")
    assert model.status == set(["running", "healthy", "active"])


async def test_health_check_embeddings(openai_server: OpenAIServerMock):
    model = Model(
        ModelConfig(
            id="test-embed",
            port=openai_server.port,
            max_requests=10,
            task="embedding",
        )
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
    result = await model._check_embedding_health()
    assert result["embedding"] == ""
    assert result["numerical"] == "no reference data"

    # 2. With verification data, correct embedding -> all ok
    model.verification_data = {"test input": expected_embedding}
    result = await model._check_embedding_health()
    assert not any(result.values())
    assert openai_server.last_request_json["input"] == ["test input"]

    # 3. Wrong embedding -> numerical fails
    openai_server.response["data"][0]["embedding"] = [0.9, 0.9, 0.9]
    assert any((await model._check_embedding_health()).values())


async def test_health_check_completions(openai_server: OpenAIServerMock):
    model = Model(
        ModelConfig(
            id="test-chat",
            port=openai_server.port,
            max_requests=10,
            task="chat",
        )
    )
    model.endpoint = openai_server.endpoint

    # 2. Simulate correct response -> healthy
    openai_server.response = {}
    assert not any((await model._check_completion_health()).values())
    assert openai_server.last_request_json["prompt"] == "2+2="

    # 2. Simulate wrong response -> unhealthy
    openai_server.response_status = 500
    assert any((await model._check_completion_health()).values())
    assert openai_server.last_request_json["prompt"] == "2+2="


async def test_health_check_streaming_tool_call_check_live(
    monkeypatch: pytest.MonkeyPatch,
):
    model = Model(
        ModelConfig(
            id="test-chat",
            port=8100,
            max_requests=10,
            task="chat",
        )
    )
    model.endpoint = "http://localhost:8100"
    assert not any(
        (await model._check_completion_streaming_tool_call_health()).values()
    )

    # offline endpoint
    model.endpoint = "http://localhost:9000"
    assert {
        "completion_tool_call": "Error connecting: Connection error."
    } == await model._check_completion_streaming_tool_call_health()


async def test_health_check_streaming_tool_call_check_mocked(
    monkeypatch: pytest.MonkeyPatch,
):
    model = Model(
        ModelConfig(
            id="test-chat",
            port=8100,
            max_requests=10,
            task="chat",
        )
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
    assert {
        "completion_tool_call": "No tool calls found",
    } == await model._check_completion_streaming_tool_call_health()

    # tool call without id returned
    call = Mock()
    call.id = None
    choice.message.tool_calls = [call]
    assert {
        "completion_tool_call": "Missing tool call ID in tool call #0: None",
    } == await model._check_completion_streaming_tool_call_health()

    # no function name
    call.id = "1234"
    call.function.name = None
    assert {
        "completion_tool_call": "Missing tool function name in tool call #0: None",
    } == await model._check_completion_streaming_tool_call_health()

    # incorrect json args
    call.id = "1234"
    call.function.name = "Asdf"
    call.function.arguments = r'{"object": 123'
    assert {
        "completion_tool_call": "Invalid argument JSON in tool call #0: '{\"object\": 123'",
    } == await model._check_completion_streaming_tool_call_health()
