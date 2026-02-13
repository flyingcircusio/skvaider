import asyncio
import base64
import json
import os
from collections.abc import AsyncGenerator, Callable, Coroutine
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Generator

import httpx
import prometheus_client
import pytest
import svcs
from argon2 import PasswordHasher
from fastapi import FastAPI
from fastapi.testclient import TestClient

import aramaki
import skvaider.auth
import skvaider.proxy.backends
import skvaider.proxy.pool
from aramaki.typing import JSONObject
from skvaider import app_factory, metrics
from skvaider.config import ModelInstanceConfig, parse_size
from skvaider.utils import TaskManager
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
    [Callable[..., Coroutine[Any, Any, bool]]],
    Callable[..., Coroutine[Any, Any, None]],
]:
    """Wait for a callable to return True.

    If AssertionErrors happen, those will be propagated if the timeout occurs
    but will be suppressed while retrying.

    """

    def decorator(
        async_condition: Callable[..., Coroutine[Any, Any, bool]],
    ) -> Callable[..., Coroutine[Any, Any, None]]:
        async def wrapped(*args: Any, **kwargs: Any) -> None:
            assertion: AssertionError | None = None

            async def loop() -> None:
                result: bool = False
                nonlocal assertion
                while True:
                    assertion = None
                    try:
                        result = await async_condition(*args, **kwargs)
                    except AssertionError as e:
                        assertion = e
                    except Exception:
                        pass

                    if result:
                        return

                    await asyncio.sleep(interval)

            try:
                await asyncio.wait_for(loop(), timeout=timeout)
            except asyncio.TimeoutError as e:
                if assertion:
                    raise assertion
                raise asyncio.TimeoutError(async_condition.__name__) from e

        return wrapped

    return decorator


@wait_for_condition()
async def backend_connection_is_up(url: str) -> bool:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{url}/manager/health")
        if resp.status_code == 200:
            return True
    return False


@svcs.fastapi.lifespan
async def test_lifespan(
    app: FastAPI, registry: svcs.Registry
) -> AsyncGenerator[None, None]:
    # This is one of the backends from the devenv.
    url = "http://127.0.0.1:8001"
    await backend_connection_is_up(url)

    backend = skvaider.proxy.backends.SkvaiderBackend(url)

    pool = skvaider.proxy.pool.Pool(
        [
            ModelInstanceConfig(
                id="gemma", instances=1, memory={"ram": parse_size("1.3G")}
            ),
            ModelInstanceConfig(
                id="embeddinggemma",
                instances=1,
                memory={"ram": parse_size("250M")},
            ),
        ],
        [backend],
    )

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

    @wait_for_condition()
    async def wait_for_models_active() -> bool:
        # Wait for at least one instance of each model to be active
        for model_id in pool.model_configs.keys():
            if pool.count_loaded_instances(model_id):
                return True
        return False

    await wait_for_models_active()

    yield
    pool.close()


@pytest.fixture
async def task_managers():
    task_managers: list[TaskManager] = []
    yield task_managers
    for tm in task_managers:
        tm.terminate()


@pytest.fixture(params=["gemma"])
def llm_model_name(request: pytest.FixtureRequest) -> str:
    result: str = request.param
    return result


@pytest.fixture
def token_db() -> Generator[DummyTokens, None, None]:
    DUMMY_TOKENS.data.clear()
    yield DUMMY_TOKENS
    DUMMY_TOKENS.data.clear()


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


@pytest.fixture(autouse=True)
def cleanup_prometheus_metrics():
    for m in metrics.__dict__.values():
        if isinstance(m, prometheus_client.metrics.MetricWrapperBase):
            m.clear()
