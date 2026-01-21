import json
from pathlib import Path

import httpx
import pytest
import svcs.fastapi
from fastapi import FastAPI
from fastapi.testclient import TestClient

from skvaider.inference import app_factory
from skvaider.inference.config import ModelConfig
from skvaider.inference.manager import Manager, Model


@pytest.fixture
def services():
    reg = svcs.Registry()
    with svcs.Container(reg) as container:
        yield container


@svcs.fastapi.lifespan
async def test_lifespan(app: FastAPI, registry: svcs.Registry):
    yield


@pytest.fixture
def client():
    with TestClient(app_factory(lifespan=test_lifespan)) as client:
        yield client


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
    m = Manager(model_config_path)
    yield m
    await m.shutdown()


@pytest.fixture
async def gemma(models_cache, model_config_path, manager):
    config = ModelConfig(
        id="gemma",
        url="https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/c90975dbd40c0c7b275fefaae758c3415c906238/gemma-3-270m-it-UD-Q4_K_XL.gguf?download=true",
        hash="e5420636e0cbfee24051ff22e9719380a3a93207a472edb18dd0c89a95f6ef80",
        context_size=4096,
        cmd_args=[],
    )

    model = Model(config)
    manager.add_model(model)

    cache = Path(".models").absolute()
    cache.mkdir(exist_ok=True)
    cache_file = cache / model.slug
    if not cache_file.exists():
        await model.download()
        model.model_file.rename(cache_file)
    model.model_file.symlink_to(cache_file)

    return model
