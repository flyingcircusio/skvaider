import asyncio
import base64
import json

import httpx
import pytest
import svcs
from argon2 import PasswordHasher
from fastapi import FastAPI
from fastapi.testclient import TestClient

import skvaider.auth
import skvaider.proxy.backends
import skvaider.proxy.pool
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


def wait_for_condition(interval=0.1, timeout=30):
    def decorator(async_condition):
        async def wrapped():
            async def loop():
                result = False
                while True:
                    try:
                        result = await async_condition()
                    except Exception:
                        pass

                    if result:
                        return

                    await asyncio.sleep(interval)

            try:
                await asyncio.wait_for(loop(), timeout=5)
            except asyncio.TimeoutError as e:
                raise asyncio.TimeoutError(async_condition.__name__) from e

        return wrapped

    return decorator


@svcs.fastapi.lifespan
async def test_lifespan(app: FastAPI, registry: svcs.Registry):
    pool = skvaider.proxy.pool.Pool()

    url = "http://127.0.0.1:8001"

    @wait_for_condition()
    async def backend_connection_is_up():
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{url}/manager/health")
            if resp.status_code == 200:
                return True

    await backend_connection_is_up()

    pool.add_backend(skvaider.proxy.backends.SkvaiderBackend(url))

    registry.register_value(skvaider.proxy.pool.Pool, pool)
    registry.register_value(skvaider.auth.AuthTokens, DUMMY_TOKENS)

    @wait_for_condition()
    async def wait_for_healthy_backends():
        return all(b.healthy for b in pool.backends)

    await wait_for_healthy_backends()

    yield {}
    pool.close()


@pytest.fixture(params=["gemma"])
def llm_model_name(request):
    return request.param


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
