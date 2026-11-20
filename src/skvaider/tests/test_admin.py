from collections.abc import Callable

import svcs
from fastapi.testclient import TestClient

from skvaider.conftest import ADMIN_TOKEN, registered_model_factory
from skvaider.proxy.backends import DummyBackend
from skvaider.proxy.models import CheckResult
from skvaider.proxy.pool import Pool
from skvaider.routers.admin import health


def summarize(check: CheckResult) -> tuple[str, str, list[object]]:
    return (
        check.name,
        check.status,
        [summarize(c) for c in check.children],
    )


async def test_health_all_ok(pool: Pool, services: svcs.Container):
    pool.backends[0].models["test-model"].checks = [
        CheckResult(name="memory", status="ok", message="ok")
    ]

    result = await health(services)

    assert summarize(result) == (
        "Skvaider",
        "ok",
        [
            (
                "test-model",
                "ok",
                [("http://test-backend", "ok", [("memory", "ok", [])])],
            )
        ],
    )


async def test_health_model_without_loaded_instance_is_critical(
    pool: Pool, services: svcs.Container
):
    pool.backends[0].models["test-model"].is_loaded = False

    result = await health(services)

    assert result.status == "critical"
    [model_check] = result.children
    assert model_check.status == "critical"
    assert model_check.message == "no loaded instance"


async def test_health_backend_check_failure_propagates(
    pool: Pool, services: svcs.Container
):
    pool.backends[0].models["test-model"].checks = [
        CheckResult(name="memory", status="warning", message="too much"),
        CheckResult(name="completion", status="ok", message="ok"),
    ]

    result = await health(services)

    assert result.status == "warning"
    [model_check] = result.children
    # The model itself is loaded, only the backend instance has problems.
    assert model_check.status == "ok"
    [backend_check] = model_check.children
    assert backend_check.status == "warning"


async def test_health_critical_wins_over_warning(
    pool: Pool,
    services: svcs.Container,
    dummy_backend_factory: Callable[..., DummyBackend],
):
    pool.backends[0].models["test-model"].checks = [
        CheckResult(name="memory", status="warning", message="too much"),
    ]
    other = dummy_backend_factory("http://other-backend")
    model = registered_model_factory("test-model", other, loaded=True)
    model.checks = [
        CheckResult(name="completion", status="critical", message="timeout")
    ]
    pool.backends.append(other)

    result = await health(services)

    assert result.status == "critical"
    [model_check] = result.children
    assert [(c.name, c.status) for c in model_check.children] == [
        ("http://test-backend", "warning"),
        ("http://other-backend", "critical"),
    ]


async def test_health_only_lists_backends_carrying_the_model(
    pool: Pool,
    services: svcs.Container,
    dummy_backend_factory: Callable[..., DummyBackend],
):
    pool.backends.append(dummy_backend_factory("http://empty-backend"))

    result = await health(services)

    [model_check] = result.children
    assert [c.name for c in model_check.children] == ["http://test-backend"]


async def test_health_ignores_models_unknown_to_the_pool(
    pool: Pool, services: svcs.Container, dummy_backend: DummyBackend
):
    registered_model_factory("unconfigured", dummy_backend)

    result = await health(services)

    assert [c.name for c in result.children] == ["test-model"]


def test_health_endpoint_requires_admin_token(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 403

    response = client.get("/health", headers={"Authorization": "Bearer wrong"})
    assert response.status_code == 401


def test_health_endpoint(client: TestClient):
    response = client.get(
        "/health", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"}
    )
    assert response.status_code == 200

    result = CheckResult.model_validate_json(response.content)
    assert result.name == "Skvaider"
    # The devenv memory figures may trigger memory warnings, which is fine here.
    assert result.status != "critical", response.text
    assert {c.name for c in result.children} == {"gemma", "embeddinggemma"}
    for model_check in result.children:
        assert model_check.status == "ok"
        assert [c.name for c in model_check.children] == [
            "http://127.0.0.1:8001"
        ]
