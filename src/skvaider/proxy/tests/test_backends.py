from collections.abc import Callable
from unittest.mock import AsyncMock

import pytest

from skvaider.config import ModelInstanceConfig
from skvaider.manifest import ManifestRequest, Serial
from skvaider.proxy.backends import (
    BackendHealthResponse,
    BackendModelInfo,
    SkvaiderBackend,
)
from skvaider.proxy.models import AIModel
from skvaider.proxy.pool import Pool
from skvaider.utils import TaskManager


def _pool_with_backend(
    backend: SkvaiderBackend,
    task_managers: list[TaskManager],
) -> Pool:
    model = AIModel(id="m1", owned_by="test", backend=backend)
    model.memory_usage = {"ram": 100}
    backend.models["m1"] = model
    backend.memory = {"ram": {"free": 1000, "total": 1000}}

    pool = Pool(
        [
            ModelInstanceConfig(
                id="m1", instances=1, memory={"ram": 100}, task="chat"
            )
        ],
        [backend],
    )
    task_managers.append(pool.tasks)
    return pool


async def test_stale_manifest_serial_triggers_reconciliation(
    task_managers: list[TaskManager],
):
    """After a backend restart its serial is behind the pool's — update_manifest
    must push the current manifest."""
    backend = SkvaiderBackend("http://inference-1")
    backend.healthy = True
    backend.backend_api = AsyncMock()

    pool = _pool_with_backend(backend, task_managers)
    await pool.rebalance()
    backend.backend_api.reset_mock()

    # backend.current_serial is still Serial.floor() — the backend hasn't
    # acknowledged the manifest yet (simulates a fresh restart).
    assert backend.current_serial < pool.map_serial

    await backend.update_manifest()

    backend.backend_api.assert_called_once()
    request = backend.backend_api.call_args.args[0]
    assert isinstance(request, ManifestRequest)
    assert request.serial == pool.map_serial
    assert "m1" in request.models


async def test_current_manifest_serial_skips_reconciliation(
    task_managers: list[TaskManager],
):
    """If the backend serial already matches the pool, no manifest is sent."""
    backend = SkvaiderBackend("http://inference-1")
    backend.healthy = True
    backend.backend_api = AsyncMock()

    pool = _pool_with_backend(backend, task_managers)
    await pool.rebalance()

    backend.current_serial = pool.map_serial
    backend.backend_api.reset_mock()

    await backend.update_manifest()

    backend.backend_api.assert_not_called()


async def test_stale_manifest_without_map_data_skips_serialization(
    task_managers: list[TaskManager],
):
    """If the backend serial is stale but no map data exists for this backend, no manifest is sent."""
    backend = SkvaiderBackend("http://inference-1")
    backend.healthy = True
    backend.backend_api = AsyncMock()

    pool = _pool_with_backend(backend, task_managers)
    await pool.rebalance()
    backend.backend_api.reset_mock()

    # backend.current_serial is still Serial.floor() — the backend hasn't
    # acknowledged the manifest yet (simulates a fresh restart).
    assert backend.current_serial < pool.map_serial

    pool.last_map = {}
    await backend.update_manifest()

    backend.backend_api.assert_not_called()


@pytest.fixture
def health_response_factory():
    def factory(
        *,
        status: set[str] | None = None,
        ram: int = 100,
        health_checks: dict[str, str] | None = None,
    ) -> BackendHealthResponse:
        return BackendHealthResponse(
            status="ok",
            current_serial=Serial.floor(),
            usage={"ram": {"free": 1000, "total": 1000}},
            models=[
                BackendModelInfo(
                    id="m1",
                    status=status if status is not None else {"active"},
                    max_requests=1,
                    memory_usage={"ram": ram},
                    health_checks=health_checks or {},
                )
            ],
        )

    return factory


async def test_update_health_records_named_checks(
    task_managers: list[TaskManager],
    health_response_factory: Callable[..., BackendHealthResponse],
):
    backend = SkvaiderBackend("http://inference-1")
    backend.healthy = True
    pool = _pool_with_backend(backend, task_managers)
    backend.backend_api = AsyncMock(
        return_value=health_response_factory(
            health_checks={"embedding": "", "completion": "timeout"}
        )
    )

    await backend._update_health()  # pyright: ignore[reportPrivateUsage]

    checks = pool.backends[0].models["m1"].checks
    assert [(c.name, c.status, c.message) for c in checks] == [
        ("memory", "ok", "ok"),
        ("embedding", "ok", "ok"),
        ("completion", "critical", "timeout"),
    ]


async def test_update_health_warns_about_memory_overuse(
    task_managers: list[TaskManager],
    health_response_factory: Callable[..., BackendHealthResponse],
):
    backend = SkvaiderBackend("http://inference-1")
    backend.healthy = True
    _pool_with_backend(backend, task_managers)
    backend.backend_api = AsyncMock(
        return_value=health_response_factory(ram=200)
    )

    await backend._update_health()  # pyright: ignore[reportPrivateUsage]

    [memory] = backend.models["m1"].checks
    assert memory.name == "memory"
    assert memory.status == "warning"
    assert memory.message == "exceeds configured memory: ram: 200 > 100"


async def test_update_health_skips_functional_checks_when_not_loaded(
    task_managers: list[TaskManager],
    health_response_factory: Callable[..., BackendHealthResponse],
):
    backend = SkvaiderBackend("http://inference-1")
    backend.healthy = True
    _pool_with_backend(backend, task_managers)
    backend.backend_api = AsyncMock(
        return_value=health_response_factory(
            status={"inactive", "stopped"},
            health_checks={"completion": "timeout"},
        )
    )

    await backend._update_health()  # pyright: ignore[reportPrivateUsage]

    assert [c.name for c in backend.models["m1"].checks] == ["memory"]
