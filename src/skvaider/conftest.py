import asyncio
import base64
import json

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
        {"gemma3": {"num_ctx": 3072}}
    )
    pool.add_backend(
        skvaider.routers.openai.Backend("http://localhost:11435", model_config)
    )
    registry.register_value(skvaider.routers.openai.Pool, pool)
    registry.register_value(skvaider.auth.AuthTokens, DUMMY_TOKENS)

    while True:
        try:
            pool.choose_backend("gemma3:1b")
        except Exception:
            await asyncio.sleep(1)
            continue
        break

    yield {}
    pool.close()


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
