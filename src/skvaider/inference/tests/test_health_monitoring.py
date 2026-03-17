from pathlib import Path
from typing import Any

import pytest

from skvaider.conftest import wait_for_condition
from skvaider.inference.config import LlamaModelFile, LlamaServerModelConfig
from skvaider.inference.conftest import ServerState, mock_llama_subprocess
from skvaider.inference.model import LlamaModel


def _make_fast_health_check_model(
    config: LlamaServerModelConfig, tmp_path: Path
) -> LlamaModel:
    """Create a Model with accelerated health-check intervals for testing."""
    model = LlamaModel(config)
    model.datadir = tmp_path
    model.health_check_interval = 0.01
    model.health_check_timeout = 0.01
    return model


@wait_for_condition()
async def is_active(model: LlamaModel, expected: bool) -> bool:
    if ("active" in model.status) == expected:
        return True
    return False


@pytest.mark.parametrize(
    "model_kwargs",
    [
        {
            "id": "test",
            "files": [LlamaModelFile(url="u", hash="h")],
        },
        {
            "id": "test-embed",
            "cmd_args": ["--embeddings"],
            "files": [LlamaModelFile(url="u", hash="h")],
        },
    ],
)
async def test_health_check(
    fake_llama_server: tuple[str, ServerState, int],
    tmp_path: Path,
    model_kwargs: dict[str, Any],
):
    url, state, port = fake_llama_server
    model = _make_fast_health_check_model(
        LlamaServerModelConfig(**model_kwargs), tmp_path
    )

    async with mock_llama_subprocess(port):
        await model.start()
        try:
            assert model.endpoint == url

            # 1. Verify it becomes healthy
            await is_active(model, True)

            # 2. Simulate failure
            state.status = 500
            await is_active(model, False)

            # 3. Simulate recovery
            state.status = 200
            await is_active(model, True)

        finally:
            await model.terminate()


async def test_health_check_embeddings(
    fake_llama_server: tuple[str, ServerState, int],
    tmp_path: Path,
):
    _, state, port = fake_llama_server

    expected_embedding = [0.1, 0.2, 0.3]
    state.body = {
        "model": "test-embed",
        "object": "list",
        "data": [
            {"embedding": expected_embedding, "index": 0, "object": "embedding"}
        ],
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }

    config = LlamaServerModelConfig(
        id="test-embed",
        cmd_args=["--embeddings"],
        files=[LlamaModelFile(url="u", hash="h")],
        context_size=1024,
        port=port,
    )

    model = _make_fast_health_check_model(config, tmp_path)
    model.verification_data = {"test input": expected_embedding}

    async with mock_llama_subprocess(port):
        await model.start()
        try:
            # 1. Verify it sends correct input and becomes healthy
            await is_active(model, True)
            assert state.last_request_json is not None
            assert state.last_request_json["input"] == ["test input"]

            # 2. Simulate wrong embedding -> unhealthy
            state.body["data"][0]["embedding"] = [0.9, 0.9, 0.9]
            await is_active(model, False)

            # 3. Correct again -> healthy
            state.body["data"][0]["embedding"] = expected_embedding
            await is_active(model, True)

        finally:
            await model.terminate()
