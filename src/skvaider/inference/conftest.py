import json
from pathlib import Path

import httpx
import pytest
import svcs.fastapi
from fastapi import FastAPI
from fastapi.testclient import TestClient

from skvaider.inference import app_factory
from skvaider.inference.manager import Manager


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
def gemma(models_cache, model_config_path, manager):
    # XXX this should be a cached gemma instance -> prepare the
    # target and avoid downloads
    filename = "gemma-3-270m-it-UD-Q4_K_XL.gguf"
    gemma_url = f"https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/{filename}?download=true"
    target = (models_cache / filename).absolute()

    if not target.exists():
        with target.open("wb") as f:
            with httpx.stream("GET", gemma_url, follow_redirects=True) as r:
                for data in r.iter_bytes():
                    f.write(data)

    config_file = model_config_path / "gemma" / "model.json"
    config_file.parent.mkdir()

    with config_file.open("w", encoding="utf-8") as f:
        config = {
            "name": "gemma",
            "filename": str(target),
            "cmd_args": [],
            "context_size": 4096,
        }
        json.dump(config, f)

    return config_file, target
