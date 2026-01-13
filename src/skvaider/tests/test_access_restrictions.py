import asyncio
import base64
import json

import pytest
import svcs
from fastapi.testclient import TestClient

from skvaider import app_factory
from skvaider.auth import AuthTokens
from skvaider.conftest import DUMMY_TOKENS, hasher
from skvaider.routers.openai import AIModel, Backend, ModelConfig, Pool


class MockBackend(Backend):
    async def monitor_health_and_update_models(self, pool):
        pool.update_model_maps()
        while True:
            await asyncio.sleep(1)


@pytest.fixture
def restricted_app_client(token_db):

    config_data = {
        "restricted-model": {"resource_groups": ["research"]},
        "public-model": {},
    }
    model_config = ModelConfig(config_data)

    pool = Pool(model_config)
    backend = MockBackend("http://mock", model_config)

    # Pre-populate backend models so they are ready when map updates
    backend.models = {
        "restricted-model": AIModel(
            id="restricted-model", owned_by="me", backend=backend
        ),
        "public-model": AIModel(
            id="public-model", owned_by="me", backend=backend
        ),
    }

    @svcs.fastapi.lifespan
    async def lifespan(app, registry):
        # Add backend here where we have a loop
        pool.add_backend(backend)
        # Give it a tiny bit of time to update the pool maps (the task runs properly now)
        await asyncio.sleep(0.01)

        registry.register_value(Pool, pool)
        registry.register_value(AuthTokens, DUMMY_TOKENS)
        yield
        pool.close()

    app = app_factory(lifespan=lifespan)
    with TestClient(app) as client:
        yield client


def create_auth_header(resource_group=None):
    secret = "secret"
    token_data = {"id": "user", "secret": secret}

    # Update the DB record which is what check_access reads
    DUMMY_TOKENS.data["user"] = {
        "secret_hash": hasher.hash(secret),
        "resource_group": resource_group,
    }

    auth_token = base64.b64encode(
        json.dumps(token_data).encode("utf-8")
    ).decode("ascii")
    return {"Authorization": f"Bearer {auth_token}"}


def test_access_restricted_model_allowed(restricted_app_client):
    headers = create_auth_header(resource_group="research")

    resp = restricted_app_client.get("/openai/v1/models", headers=headers)
    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()["data"]]
    assert "restricted-model" in ids

    resp = restricted_app_client.get(
        "/openai/v1/models/restricted-model", headers=headers
    )
    assert resp.status_code == 200


def test_access_restricted_model_denied(restricted_app_client):
    headers = create_auth_header(resource_group="marketing")

    resp = restricted_app_client.get("/openai/v1/models", headers=headers)
    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()["data"]]
    assert "restricted-model" not in ids
    assert "public-model" in ids

    resp = restricted_app_client.get(
        "/openai/v1/models/restricted-model", headers=headers
    )
    assert resp.status_code == 403


def test_access_public_model(restricted_app_client):
    headers = create_auth_header(resource_group="marketing")
    resp = restricted_app_client.get(
        "/openai/v1/models/public-model", headers=headers
    )
    assert resp.status_code == 200


def test_access_restricted_model_no_resource_group(restricted_app_client):
    headers = create_auth_header(resource_group=None)

    resp = restricted_app_client.get("/openai/v1/models", headers=headers)
    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()["data"]]
    assert "restricted-model" not in ids

    resp = restricted_app_client.get(
        "/openai/v1/models/restricted-model", headers=headers
    )
    assert resp.status_code == 403
