import asyncio
import json
import os
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from skvaider.inference.main import app
from skvaider.inference.manager import ModelConfig, ModelManager, RunningModel

client = TestClient(app)


@pytest.fixture
def model_config_path(tmp_path):
    p = tmp_path / "models"
    p.mkdir()
    return p


@pytest.fixture
def models_cache():
    cache_dir = Path(".models")
    if not cache_dir.exists():
        cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def gemma(models_cache, model_config_path):
    filename = "gemma-3-270m-it-UD-IQ2_XXS.gguf"
    gemma_url = f"https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/{filename}?download=true"
    target = (models_cache / filename).absolute()

    if not target.exists():
        with target.open("wb") as f:
            with httpx.stream("GET", gemma_url, follow_redirects=True) as r:
                for data in r.iter_bytes():
                    f.write(data)

    config_file = model_config_path / "gemma.json"
    with config_file.open("w", encoding="utf-8") as f:
        config = {
            "name": "gemma",
            "filename": str(target),
            "cmd_args": [],
            "context_size": 4096,
        }
        json.dump(config, f)

    return config_file, target


@pytest.mark.asyncio
async def test_manager_get_config(clean_models_dir):
    manager = ModelManager()

    # Create a dummy metadata file
    meta = {"name": "test-model", "cmd_args": ["-t", "4"], "context_size": 1024}
    with open("models/test_file.json", "w") as f:
        json.dump(meta, f)

    config = await manager.get_model_config("test-model")
    assert config is not None
    assert config.name == "test-model"
    assert config.filename == "test_file"
    assert config.cmd_args == ["-t", "4"]
    assert config.context_size == 1024


@pytest.mark.asyncio
async def test_manager_start_model(gemma, model_config_path):
    manager = ModelManager(model_config_path)

    with pytest.raises(KeyError):
        await manager.get_or_start_model("unknown-model")

    model = await manager.get_or_start_model("gemma")
    assert model.config.name == "gemma"
    assert model.endpoint

    async with httpx.AsyncClient() as client:
        # Check health
        r = await client.get(f"{model.endpoint}/health")
        r.raise_for_status()
        assert r.json() == {"status": "ok"}

        # Get model info via OpenAI-compatible endpoint
        r = await client.get(f"{model.endpoint}/v1/models")
        r.raise_for_status()
        models = r.json()
        del models["data"][0]["created"]
        assert models == {
            "data": [
                {
                    "id": str(gemma[1]),
                    "meta": {
                        "n_ctx_train": 32768,
                        "n_embd": 640,
                        "n_params": 268098176,
                        "n_vocab": 262144,
                        "size": 173576704,
                        "vocab_type": 1,
                    },
                    "object": "model",
                    "owned_by": "llamacpp",
                },
            ],
            "object": "list",
        }

        # Run a simple completion via OpenAI-compatible chat API
        r = await client.post(
            f"{model.endpoint}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Say hello world"}],
                "max_tokens": 1000,
            },
        )
        r.raise_for_status()
        chat = r.json()
        assert "choices" in chat
        assert len(chat["choices"]) > 0
        assert "message" in chat["choices"][0]
        content = chat["choices"][0]["message"]["content"]
        assert len(content) > 0  # Got some response

    await manager.unload_model("gemma")


@pytest.mark.asyncio
async def test_load_model(gemma):
    # Setup metadata
    meta = {
        "name": "test-model",
    }
    with open("models/test_file.json", "w") as f:
        json.dump(meta, f)

    # Mock manager in main
    mock_running_model = MagicMock()
    mock_running_model.port = 8080

    with patch(
        "skvaider.inference.main.manager.get_or_start_model",
        return_value=mock_running_model,
    ) as mock_get_model:

        response = client.post(
            "/get_running_model_or_load", json={"model": "test-model"}
        )

        assert response.status_code == 200
        assert response.json() == {"port": 8080}

        mock_get_model.assert_called_with("test-model")


@pytest.mark.asyncio
async def test_unload_model_manager(clean_models_dir):
    manager = ModelManager()

    # Mock a running model
    config = ModelConfig(name="test-model", filename="test_file")
    running_model = RunningModel(config, clean_models_dir)
    running_model.process = mock_proc = AsyncMock()
    mock_proc.terminate = MagicMock()
    mock_proc.wait = AsyncMock()

    manager.running_models["test-model"] = running_model

    await manager.unload_model("test-model")

    assert "test-model" not in manager.running_models
    mock_proc.terminate.assert_called_once()
    mock_proc.wait.assert_called_once()


@pytest.mark.asyncio
async def test_unload_model_api(clean_models_dir):
    # We need to patch the global manager instance used by the router
    with patch("skvaider.inference.routers.models.manager") as mock_manager:
        mock_manager.unload_model = AsyncMock()

        response = client.post("/unload", json={"model": "test-model"})

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        mock_manager.unload_model.assert_called_once_with("test-model")


@pytest.mark.asyncio
async def test_unload_model_api_missing_model(clean_models_dir):
    response = client.post("/unload", json={})
    assert response.status_code == 400
