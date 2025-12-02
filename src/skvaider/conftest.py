import asyncio
import base64
import json
import os
from pathlib import Path

import httpx
import pytest
import svcs
from argon2 import PasswordHasher
from fastapi import FastAPI
from fastapi.testclient import TestClient

import skvaider.routers.openai
from skvaider import app_factory

hasher = PasswordHasher()


class DummyTokens:
    def __init__(self):
        self.data = {}

    async def get(self, key):
        return self.data.get(key)

    async def keys(self):
        return self.data.keys()


DUMMY_TOKENS = DummyTokens()


@pytest.fixture
def services():
    reg = svcs.Registry()
    reg.register_value(skvaider.auth.AuthTokens, DUMMY_TOKENS)
    with svcs.Container(reg) as container:
        yield container


@svcs.fastapi.lifespan
async def test_lifespan(app: FastAPI, registry: svcs.Registry):
    pool = skvaider.routers.openai.Pool()
    model_config = skvaider.routers.openai.ModelConfig(
        {"TinyMistral-248M-v2-Instruct": {"num_ctx": 2048}}
    )

    from skvaider.inference.manager import ModelManager

    manager = ModelManager()

    # Ensure model exists
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_name = "TinyMistral-248M-v2-Instruct"
    filename = "TinyMistral-248M-v2-Instruct.Q2_K.gguf"
    model_path = models_dir / filename
    json_path = models_dir / f"{filename}.json"

    if not model_path.exists():
        url = "https://huggingface.co/M4-ai/TinyMistral-248M-v2-Instruct-GGUF/resolve/main/TinyMistral-248M-v2-Instruct.Q2_K.gguf?download=true"
        async with httpx.AsyncClient(follow_redirects=True) as client:
            async with client.stream("GET", url) as response:
                with open(model_path, "wb") as f:
                    async for chunk in response.aiter_bytes():
                        f.write(chunk)

    if not json_path.exists():
        with open(json_path, "w") as f:
            json.dump(
                {
                    "name": model_name,
                    "context_size": 2048,
                    "cmd_args": ["--embedding", "--pooling", "mean"],
                },
                f,
            )

    pool.add_backend(
        skvaider.routers.openai.SkvaiderBackend(
            "http://localhost:0", model_config, manager
        )
    )
    registry.register_value(skvaider.routers.openai.Pool, pool)
    registry.register_value(skvaider.auth.AuthTokens, DUMMY_TOKENS)

    yield {}
    pool.close()
    for model in manager.running_models.values():
        try:
            model.process.terminate()
            await model.process.wait()
        except Exception:
            pass


@pytest.fixture
def token_db():
    DUMMY_TOKENS.data.clear()
    yield DUMMY_TOKENS


@pytest.fixture
async def auth_token(token_db):
    """Return a valid auth token."""
    secret = "asdf"
    token_db.data["user"] = {"secret_hash": hasher.hash(secret)}
    auth_token = base64.b64encode(
        json.dumps({"id": "user", "secret": secret}).encode("utf-8")
    )
    yield auth_token.decode("ascii")


@pytest.fixture
async def auth_header(client, auth_token):
    """Inject a valid auth header into all client requests."""
    header = {"Authorization": f"Bearer {auth_token}"}
    client.headers.update(header)
    yield


@pytest.fixture
def client():
    with TestClient(app_factory(lifespan=test_lifespan)) as client:
        yield client
