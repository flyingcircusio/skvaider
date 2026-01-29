import asyncio
from pathlib import Path

import httpx
import pytest

from skvaider.inference.config import ModelConfig, ModelFile
from skvaider.inference.manager import Manager, Model


async def test_manager_start_crash_quick_return(gemma: Model, manager: Manager):
    gemma.config.cmd_args = ["--asdf"]
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(manager.get_or_start_model("gemma"), timeout=10)


async def test_download_model_success(gemma: Model):
    await gemma.download()
    assert gemma.model_files[0].exists()
    assert gemma.integrity_marker_file.exists()


async def test_download_model_wrong_hash(tmp_path: Path, gguf_http_server: str):
    config = ModelConfig(
        id="gemma",
        files=[
            ModelFile(
                url=f"{gguf_http_server}/not-a-model.gguf",
                hash="foobar",
            )
        ],
    )
    model = Model(config)
    model.datadir = tmp_path
    with pytest.raises(ValueError) as e:
        await model.download()
    assert (
        e.value.args[0]
        == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    assert model.model_files[0].exists()
    assert not model.integrity_marker_file.exists()


async def test_manager_start_model(gemma: Model, manager: Manager):
    with pytest.raises(KeyError):
        await manager.get_or_start_model("unknown-model")

    model = await manager.get_or_start_model("gemma")
    assert model.config.id == "gemma"
    assert model.endpoint
    assert model.endpoint.startswith("http://127.0.0.1:")

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

    assert "active" in model.status
    assert model.process_status == "running"
    assert model.health_status == "healthy"

    await manager.unload_model("gemma")

    assert "active" not in model.status
    assert model.process_status == "stopped"
    assert model.health_status == ""


async def test_download_split_model(tmp_path: Path, gguf_http_server: str):
    config = ModelConfig(
        id="split-gemma",
        files=[
            ModelFile(
                url=f"{gguf_http_server}/split-gguf-1.gguf",
                hash="4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865",
            ),
            ModelFile(
                url=f"{gguf_http_server}/split-gguf-2.gguf",
                hash="53c234e5e8472b6ac51c1ae1cab3fe06fad053beb8ebfd8977b010655bfdd3c3",
            ),
        ],
    )
    model = Model(config)
    model.datadir = tmp_path
    await model.download()
    assert model.model_files[0].exists()
    assert model.model_files[1].exists()
    assert model.integrity_marker_file.exists()
