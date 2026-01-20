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
async def manager(model_config_path):
    m = ModelManager(model_config_path)
    yield m
    await m.shutdown()


@pytest.fixture
def gemma(models_cache, model_config_path):
    filename = "gemma-3-270m-it-UD-Q4_K_XL.gguf"
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
async def test_manager_get_config(manager):
    meta = {
        "name": "test-model",
        "cmd_args": ["-t", "4"],
        "filename": "test_file",
        "context_size": 1024,
    }
    with (manager.models_dir / "test_file.json").open("w") as f:
        json.dump(meta, f)

    config = await manager.get_model_config("test-model")
    assert config is not None
    assert config.name == "test-model"
    assert config.filename == "test_file"
    assert config.cmd_args == ["-t", "4"]
    assert config.context_size == 1024


@pytest.mark.asyncio
async def test_manager_start_crash_quick_return(gemma, model_config_path):
    manager = ModelManager(model_config_path)

    config = json.loads(gemma[0].read_text())
    config["cmd_args"] = ["--asdf"]
    gemma[0].write_text(json.dumps(config))

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(manager.get_or_start_model("gemma"), timeout=5)


@pytest.mark.asyncio
async def test_manager_start_model(gemma, manager):
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
                    "id": "gemma",
                    "meta": {
                        "n_ctx_train": 32768,
                        "n_embd": 640,
                        "n_params": 268098176,
                        "n_vocab": 262144,
                        "size": 247407104,
                        "vocab_type": 1,
                    },
                    "object": "model",
                    "owned_by": "llamacpp",
                },
            ],
            "models": [
                {
                    "capabilities": ["completion"],
                    "description": "",
                    "details": {
                        "families": [""],
                        "family": "",
                        "format": "gguf",
                        "parameter_size": "",
                        "parent_model": "",
                        "quantization_level": "",
                    },
                    "digest": "",
                    "model": "gemma",
                    "modified_at": "",
                    "name": "gemma",
                    "parameters": "",
                    "size": "",
                    "tags": [""],
                    "type": "model",
                }
            ],
            "object": "list",
        }

        # Run a simple completion via OpenAI-compatible chat API
        r = await client.post(
            f"{model.endpoint}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "generate 5 numbers"}],
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

    assert "gemma" not in manager.running_models

    assert model._shutdown is True
