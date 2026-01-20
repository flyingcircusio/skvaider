import asyncio
import json

import httpx
import pytest

from skvaider.inference.config import ModelConfig
from skvaider.inference.manager import Manager, Model


async def test_manager_get_config(manager):
    meta = {
        "name": "test-model",
        "cmd_args": ["-t", "4"],
        "filename": "test_file",
        "context_size": 1024,
    }
    (manager.models_dir / "test").mkdir()
    with (manager.models_dir / "test" / "test_file.json").open("w") as f:
        json.dump(meta, f)

    config = await manager.get_model_config("test-model")
    assert config is not None
    assert config.name == "test-model"
    assert config.filename == "test_file"
    assert config.cmd_args == ["-t", "4"]
    assert config.context_size == 1024


async def test_manager_start_crash_quick_return(gemma, model_config_path):
    manager = Manager(model_config_path)

    config = json.loads(gemma[0].read_text())
    config["cmd_args"] = ["--asdf"]
    gemma[0].write_text(json.dumps(config))

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(manager.get_or_start_model("gemma"), timeout=5)


async def test_download_model_success(tmp_path):
    config = ModelConfig(
        id="gemma",
        url="https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-UD-Q4_K_XL.gguf?download=true",
        hash="e5420636e0cbfee24051ff22e9719380a3a93207a472edb18dd0c89a95f6ef80",
    )
    model = Model(config)
    model.datadir = tmp_path
    await model.download()
    assert model.model_file.exists()
    assert model.integrity_marker_file.exists()


async def test_download_model_wrong_hash(tmp_path):
    config = ModelConfig(
        id="gemma",
        url="https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-UD-Q4_K_XL.gguf?download=true",
        hash="foobar",
    )
    model = Model(config)
    model.datadir = tmp_path
    with pytest.raises(ValueError) as e:
        await model.download()
    assert (
        e.value.args[0]
        == "e5420636e0cbfee24051ff22e9719380a3a93207a472edb18dd0c89a95f6ef80"
    )
    assert model.model_file.exists()
    assert not model.integrity_marker_file.exists()


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
