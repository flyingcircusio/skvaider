import asyncio
import base64
import json
import os
from collections.abc import AsyncGenerator, Callable, Coroutine
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Generator, Any


import httpx
import pytest
import svcs
from argon2 import PasswordHasher
from fastapi import FastAPI
from fastapi.testclient import TestClient

import aramaki
import skvaider.auth
import skvaider.proxy.backends
import skvaider.proxy.pool
import skvaider.routers.openai
from skvaider import app_factory

hasher = PasswordHasher()


class DummyTokens(aramaki.AbstractCollection):
    collection = "test.tokens"

    def __init__(self):
        self.data: dict[str, JSONObject] = {}

    async def get(
        self, key: str, default: JSONObject | None = None
    ) -> JSONObject | None:
        return self.data.get(key, default)

    async def keys(self) -> list[str]:
        return list(self.data.keys())

    @asynccontextmanager
    async def get_collection_with_session(
        self,
    ) -> AsyncGenerator["DummyTokens"]:
        yield self


DUMMY_TOKENS = DummyTokens()


@pytest.fixture
def svcs_registry():
    yield svcs.Registry()


@pytest.fixture
def services(
    svcs_registry: svcs.Registry,
) -> Generator[svcs.Container, None, None]:
    svcs_registry.register_factory(  # pyright: ignore[reportUnknownMemberType]
        skvaider.auth.AuthTokens,
        DUMMY_TOKENS.get_collection_with_session,
        enter=False,
    )
    with svcs.Container(svcs_registry) as container:
        yield container


def wait_for_condition(interval: float = 0.1, timeout: float = 30) -> Callable[
    [Callable[[], Coroutine[Any, Any, bool]]],
    Callable[[], Coroutine[Any, Any, None]],
]:
    def decorator(
        async_condition: Callable[[], Coroutine[Any, Any, bool]],
    ) -> Callable[[], Coroutine[Any, Any, None]]:
        async def wrapped() -> None:
            async def loop() -> None:
                result: bool = False
                while True:
                    try:
                        result = await async_condition()
                    except Exception:
                        pass

                    if result:
                        return

                    await asyncio.sleep(interval)

            try:
                await asyncio.wait_for(loop(), timeout=timeout)
            except asyncio.TimeoutError as e:
                raise asyncio.TimeoutError(async_condition.__name__) from e

        return wrapped

    return decorator


@svcs.fastapi.lifespan
async def test_lifespan(
    app: FastAPI, registry: svcs.Registry
) -> AsyncGenerator[None, None]:
    pool = skvaider.proxy.pool.Pool()

    url = "http://127.0.0.1:8001"

    @wait_for_condition()
    async def backend_connection_is_up() -> bool:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{url}/manager/health")
            if resp.status_code == 200:
                return True
        return False

    await backend_connection_is_up()

    pool.add_backend(skvaider.proxy.backends.SkvaiderBackend(url))

    registry.register_value(  # pyright: ignore[reportUnknownMemberType]
        skvaider.proxy.pool.Pool, pool
    )
    registry.register_factory(  # pyright: ignore[reportUnknownMemberType]
        skvaider.auth.AuthTokens,
        DUMMY_TOKENS.get_collection_with_session,
        enter=False,
    )

    @wait_for_condition()
    async def wait_for_healthy_backends() -> bool:
        return all(b.healthy for b in pool.backends)

    await wait_for_healthy_backends()

    yield
    pool.close()


@pytest.fixture(params=["gemma"])
def llm_model_name(request: pytest.FixtureRequest) -> str:
    result: str = request.param
    return result


@pytest.fixture
def token_db() -> Generator[DummyTokens, None, None]:
    DUMMY_TOKENS.data.clear()
    yield DUMMY_TOKENS


@pytest.fixture
async def auth_token(token_db: DummyTokens) -> AsyncGenerator[str, None]:
    """Return a valid auth token."""
    secret = "asdf"
    token_db.data["user"] = {"secret_hash": hasher.hash(secret)}
    auth_token = base64.b64encode(
        json.dumps({"id": "user", "secret": secret}).encode("utf-8")
    )
    yield auth_token.decode("ascii")


@pytest.fixture
async def auth_header(
    client: TestClient, auth_token: str
) -> AsyncGenerator[None, None]:
    """Inject a valid auth header into all client requests."""
    header = {"Authorization": f"Bearer {auth_token}"}
    client.headers.update(header)
    yield


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    with TestClient(app_factory(lifespan=test_lifespan)) as client:
        yield client
