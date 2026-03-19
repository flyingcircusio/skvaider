import asyncio

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
            context_size=1024,
            max_requests=10,
            port=1000,
            task="chat",
        )
    )
    model.health_check_interval = 0.01
    model.health_check_timeout = 0.01

    async def health_ok() -> bool:
        return True

    async def health_not_ok() -> bool:
        return False

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
            context_size=1024,
            port=openai_server.port,
            max_requests=10,
            task="embedding",
        )
    )
    model.endpoint = openai_server.endpoint

    expected_embedding = [0.1, 0.2, 0.3]
    model.verification_data = {"test input": expected_embedding}

    openai_server.response = {
        "model": "test-embed",
        "object": "list",
        "data": [
            {"embedding": expected_embedding, "index": 0, "object": "embedding"}
        ],
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }
    assert await model._check_embedding_health()
    assert openai_server.last_request_json["input"] == ["test input"]

    # 2. Simulate wrong embedding -> unhealthy
    openai_server.response["data"][0]["embedding"] = [0.9, 0.9, 0.9]
    assert not await model._check_embedding_health()


async def test_health_check_completions(openai_server: OpenAIServerMock):
    model = Model(
        ModelConfig(
            id="test-embed",
            context_size=1024,
            port=openai_server.port,
            max_requests=10,
            task="chat",
        )
    )
    model.endpoint = openai_server.endpoint

    # 2. Simulate correct response -> healthy
    openai_server.response = {}
    assert await model._check_completion_health()
    assert openai_server.last_request_json["prompt"] == "2+2="

    # 2. Simulate wrong response -> unhealthy
    openai_server.response_status = 500
    assert not await model._check_completion_health()
    assert openai_server.last_request_json["prompt"] == "2+2="
