import asyncio
import json
import os
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from inference.main import app
from inference.manager import ModelConfig, ModelManager, RunningModel

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
        b"HTTP server is listening, hostname: 127.0.0.1, port: 8555, http threads: 9\n",
        b"",
    ]
    # Mock stdout to be empty
    mock_proc.stdout.readline.return_value = b""

    with (
        patch(
            "asyncio.create_subprocess_exec", return_value=mock_proc
        ) as mock_exec,
        patch(
            "inference.manager.ModelManager._wait_for_startup",
            return_value=None,
        ),
    ):

        model = await manager.get_or_start_model("test-model")

        assert model is not None
        assert model.config.name == "test-model"
        assert model.port == 8555

        mock_exec.assert_called_once()
        args = mock_exec.call_args[0]
        assert args[0] == "llama-server"
        assert "-m" in args
        assert str(Path("models/test_file")) in args
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
        "inference.main.manager.get_or_start_model",
        return_value=mock_running_model,
    ) as mock_get_model:

        response = client.post("/load", json={"model": "test-model"})

        assert response.status_code == 200
        assert response.json() == {"port": 8080}

        mock_get_model.assert_called_with("test-model")
