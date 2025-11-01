import asyncio
import base64
import json

import pytest
import svcs
from argon2 import PasswordHasher
from fastapi import FastAPI
from fastapi.testclient import TestClient

import skvaider.auth
import skvaider.routers.openai
from aramaki.collection import Record
from aramaki.manager import Manager
from skvaider import app_factory

hasher = PasswordHasher()


@pytest.fixture
async def services(tmpdir):
    reg = svcs.Registry()

    aramaki = Manager(
        "noprincipal", "noapplication", "nourl", "nosecret", tmpdir
    )

    auth_tokens = aramaki.register_collection(skvaider.auth.AuthTokens)
    reg.register_value(skvaider.auth.AuthTokens, auth_tokens.bound_collection)

    with svcs.Container(reg) as container:
        yield container


@pytest.fixture
def lifespan(tmpdir, auth_token):
    @svcs.fastapi.lifespan
    async def test_lifespan(app: FastAPI, registry: svcs.Registry):
        pool = skvaider.routers.openai.Pool()
        model_config = skvaider.routers.openai.ModelConfig(
            {"gemma3": {"num_ctx": 3072}}
        )
        pool.add_backend(
            skvaider.routers.openai.Backend(
                "http://localhost:11435", model_config
            )
        )
        registry.register_value(skvaider.routers.openai.Pool, pool)
        aramaki = Manager(
            "noprincipal", "noapplication", "nourl", "nosecret", tmpdir
        )
        auth_tokens = aramaki.register_collection(skvaider.auth.AuthTokens)

        async with aramaki.db.session() as session:
            await Record.create(
                session,
                collection=auth_tokens.collection.collection,
                partition="p1",
                record_id="user",
                version="1",
                data=dict(secret_hash=auth_token[0]),
            )

        registry.register_value(
            skvaider.auth.AuthTokens, auth_tokens.bound_collection
        )

        tries = 10
        while tries := tries - 1:
            if "gemma3:1b" in pool.models:
                break
            await asyncio.sleep(1)
        else:
            raise ValueError("Missing sample model")

        yield {}
        pool.close()

    yield test_lifespan


@pytest.fixture
async def auth_token():
    """Return a valid auth token."""
    secret = "asdf"
    secret_hash = hasher.hash(secret)
    auth_key = base64.b64encode(
        json.dumps({"id": "user", "secret": secret}).encode("utf-8")
    ).decode("ascii)")
    yield secret_hash, auth_key


@pytest.fixture
async def auth_header(client, auth_token):
    """Inject a valid auth header into all client requests."""
    header = {"Authorization": f"Bearer {auth_token[1]}"}
    client.headers.update(header)
    yield


@pytest.fixture
def client(lifespan):
    with TestClient(app_factory(lifespan=lifespan)) as client:
        yield client
