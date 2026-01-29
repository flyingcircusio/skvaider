from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from skvaider.inference.config import ModelConfig, ModelFile
from skvaider.inference.manager import Model


@pytest.mark.asyncio
async def test_inference_endpoints_normalize_model_name(
    client: TestClient, gemma: Model
):
    """
    Test that inference endpoints normalize model name to lowercase.
    """
    model_id = gemma.config.id  # should be "gemma"
    mixed_case_id = model_id.upper()

    # Test /models/{model_name}
    response = client.get(f"/models/{mixed_case_id}")
    assert response.status_code == 200
    assert response.json()["id"] == model_id

    # Test /models/{model_name}/load
    response = client.post(f"/models/{mixed_case_id}/load")
    assert response.status_code == 200

    # Test /models/{model_name}/unload
    response = client.post(f"/models/{mixed_case_id}/unload")
    assert response.status_code == 200


def test_model_config_normalizes_id_to_lowercase():
    """
    Test that ModelConfig normalizes the ID to lowercase.
    """
    config = ModelConfig(
        id="Gemma-2-2b",
        files=[
            ModelFile(
                url="https://example.com/model.gguf",
                hash="sha256:1234567890abcdef",
            )
        ],
        llama_server=Path("llama-server"),
    )
    assert config.id == "gemma-2-2b"
