from pathlib import Path

import httpx
import pytest

from skvaider.inference.config import LlamaModelFile, LlamaServerModelConfig
from skvaider.inference.conftest import TestAPI
from skvaider.inference.model import Model
from skvaider.manifest import ManifestRequest, Serial
from skvaider.proxy.backends import BackendHealthRequest


async def test_inference_endpoints_normalize_model_name(
    test_api: TestAPI, gemma: Model
):
    """
    Test that inference endpoints normalize model name to lowercase.
    """
    model_id = gemma.config.id
    assert model_id == "gemma"

    health = await test_api(BackendHealthRequest())
    assert model_id in {m.id for m in health.models}

    await test_api(
        ManifestRequest(
            models={"GemmA"}, serial=Serial(generation="1", serial=1)
        )
    )

    # unknown model name is rejected
    with pytest.raises(httpx.HTTPStatusError):
        await test_api(
            ManifestRequest(
                models={"Gem-ma"}, serial=Serial(generation="1", serial=1)
            )
        )

    await test_api(
        ManifestRequest(models=set(), serial=Serial(generation="1", serial=2))
    )


def test_model_config_normalizes_id_to_lowercase():
    """
    Test that LlamaServerModelConfig normalizes the ID to lowercase.
    """
    config = LlamaServerModelConfig(
        id="Gemma-2-2b",
        files=[
            LlamaModelFile(
                url="https://example.com/model.gguf",
                hash="sha256:1234567890abcdef",
            )
        ],
        llama_server=Path("llama-server"),
        context_size=1024,
        port=0,
        task="chat",
    )
    assert config.id == "gemma-2-2b"
