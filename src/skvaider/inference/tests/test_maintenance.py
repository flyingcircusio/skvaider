from collections.abc import AsyncGenerator, Generator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import cast

import pytest
import svcs
import svcs.fastapi
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.types import Scope

from skvaider.inference import app_factory
from skvaider.inference.config import (
    Config,
    LoggingConfig,
    OpenAIConfig,
    ServerConfig,
)
from skvaider.inference.maintenance import Maintenance
from skvaider.inference.manager import Manager
from skvaider.inference.model import Model
from skvaider.inference.routers.manager import require_local
from skvaider.manifest import Serial


def fake_model(model_id: str, process_status: str) -> Model:
    return cast(
        Model,
        SimpleNamespace(
            config=SimpleNamespace(id=model_id),
            process_status=process_status,
        ),
    )


def request_from(host: str | None) -> Request:
    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": "/manager/maintenance",
        "query_string": b"",
        "headers": [],
        "server": ("testserver", 80),
        "scheme": "http",
    }
    if host is not None:
        scope["client"] = (host, 50000)
    return Request(scope)


def test_local_guard_rejects_remote_clients():
    with pytest.raises(HTTPException) as e:
        require_local(request_from("10.0.0.1"))
    assert e.value.status_code == 403

    for local in ("127.0.0.1", "::1", "::ffff:127.0.0.1"):
        require_local(request_from(local))

    # No client information at all (e.g. a unix socket) is not local.
    with pytest.raises(HTTPException):
        require_local(request_from(None))


def test_maintenance_phase_lifecycle():
    maintenance = Maintenance()
    models: list[Model] = [fake_model("gemma", "running")]
    serial = Serial(generation="00000-00001")

    assert maintenance.phase(set(), serial, models) == "none"

    maintenance.request("reboot")
    # Without any contact from the proxy we cannot know whether our models
    # were moved away, so a fresh server must not claim it is drained.
    assert maintenance.phase(set(), Serial.floor(), models) == "requested"
    # The proxy still asks us to run the model: it has not reacted yet.
    assert maintenance.phase({"gemma"}, serial, models) == "requested"

    # The proxy removed us from its placement map; the model still runs until
    # the in-flight requests are done.
    status = maintenance.status(set(), serial, models)
    assert status.phase == "draining"
    assert status.pending_models == ["gemma"]
    assert status.manifest == []

    models[0].process_status = "stopped"
    status = maintenance.status(set(), serial, models)
    assert status.phase == "drained"
    assert status.pending_models == []


def test_maintenance_request_is_idempotent():
    maintenance = Maintenance()
    maintenance.request("first")
    since = maintenance.since
    maintenance.request("second")
    assert maintenance.since == since
    assert maintenance.reason == "second"

    maintenance.leave()
    maintenance.leave()
    assert maintenance.requested is False
    assert maintenance.since is None
    assert maintenance.reason is None


async def test_manager_reports_maintenance(manager: Manager):
    assert manager.maintenance_status().phase == "none"

    status = manager.request_maintenance("reboot")
    assert status.phase == "requested"
    assert status.since is not None

    assert manager.leave_maintenance().phase == "none"


@contextmanager
def client_for(
    manager: Manager, host: str = "testclient"
) -> Generator[TestClient, None, None]:
    """Run the inference app against the test manager from a given peer."""

    @svcs.fastapi.lifespan
    async def lifespan(
        app: object, registry: svcs.Registry
    ) -> AsyncGenerator[None, None]:
        registry.register_value(  # pyright: ignore[reportUnknownMemberType]
            Manager, manager
        )
        yield

    config = Config(
        models_dir=manager.models_dir,
        server=ServerConfig(),
        logging=LoggingConfig(),
        openai=OpenAIConfig(models=[]),
    )
    with TestClient(
        app_factory(config=config, lifespan=lifespan),
        client=(host, 50000),
    ) as client:
        yield client


async def test_maintenance_endpoints_are_local_only(manager: Manager):
    with client_for(manager) as client:
        # Reading the state is fine from anywhere.
        r = client.get("/manager/maintenance")
        assert r.status_code == 200
        assert r.json()["phase"] == "none"

        r = client.post("/manager/maintenance", json={"reason": "reboot"})
        assert r.status_code == 403
        assert manager.maintenance_status().phase == "none"

        r = client.delete("/manager/maintenance")
        assert r.status_code == 403


async def test_maintenance_endpoints(manager: Manager):
    with client_for(manager, host="127.0.0.1") as client:
        r = client.post("/manager/maintenance", json={"reason": "reboot"})
        assert r.status_code == 200
        assert r.json()["phase"] == "requested"
        assert r.json()["reason"] == "reboot"

        # The proxy sees the request in the health response.
        r = client.get("/manager/health")
        assert r.status_code == 200
        assert r.json()["maintenance"] == "requested"

        r = client.delete("/manager/maintenance")
        assert r.status_code == 200
        assert r.json()["phase"] == "none"

        r = client.get("/manager/health")
        assert r.json()["maintenance"] == "none"
