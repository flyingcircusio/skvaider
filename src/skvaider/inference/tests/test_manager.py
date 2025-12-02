import asyncio
import json
import os
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from skvaider.inference.main import app
from skvaider.inference.manager import ModelConfig, ModelManager, RunningModel

client = TestClient(app)


@pytest.fixture
def clean_models_dir():
    if os.path.exists("models"):
        shutil.rmtree("models")
    os.makedirs("models", exist_ok=True)
    yield
    if os.path.exists("models"):
        shutil.rmtree("models")


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
async def test_manager_start_model(clean_models_dir):
    manager = ModelManager()

    # Create a dummy metadata file
    meta = {
        "name": "test-model",
        "cmd_args": [],
    }
    with open("models/test_file.json", "w") as f:
        json.dump(meta, f)

    # Mock subprocess and httpx
    mock_proc = AsyncMock()
    mock_proc.kill = MagicMock()

    # Mock stderr to return the port line
    mock_proc.stderr.readline.side_effect = [
        b"some log line\n",
        b"main: HTTP server is listening, hostname: 127.0.0.1, port: 62550, http threads: 9\n",
        b"",
    ]
    # Mock stdout to be empty
    mock_proc.stdout.readline.return_value = b""

    with (
        patch(
            "asyncio.create_subprocess_exec", return_value=mock_proc
        ) as mock_exec,
        patch(
            "skvaider.inference.manager.ModelManager._wait_for_startup",
            return_value=None,
        ),
    ):

        model = await manager.get_or_start_model("test-model")

        assert model is not None
        assert model.config.name == "test-model"
        assert model.port == 62550

        mock_exec.assert_called_once()
        args = mock_exec.call_args[0]
        assert args[0] == "llama-server"
        assert "--model" in args
        assert "models/test_file" in args
        assert "--port" in args
        assert args[args.index("--port") + 1] == "0"


@pytest.mark.asyncio
async def test_load_model(clean_models_dir):
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

        response = client.post("/load", json={"model": "test-model"})

        assert response.status_code == 200
        assert response.json() == {"port": 8080}

        mock_get_model.assert_called_with("test-model")


@pytest.mark.asyncio
async def test_unload_model_manager(clean_models_dir):
    manager = ModelManager()

    # Mock a running model
    config = ModelConfig(name="test-model", filename="test_file")
    mock_proc = AsyncMock()
    mock_proc.terminate = MagicMock()
    mock_proc.wait = AsyncMock()

    running_model = RunningModel(config, mock_proc, 8080)
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
