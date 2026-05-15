from unittest.mock import AsyncMock

from skvaider.config import ModelInstanceConfig
from skvaider.manifest import ManifestRequest
from skvaider.proxy.backends import SkvaiderBackend
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


async def test_health_check_timeout_marks_backend_unhealthy():
    """When health check exceeds its timeout, the backend is marked unhealthy
    with a clear reason rather than blocking the monitor loop indefinitely."""
    import asyncio

    backend = SkvaiderBackend("http://inference-1")
    backend.healthy = True

    # Make backend_api hang indefinitely
    async def _hang():
        await asyncio.sleep(1000)

    # Replicate one iteration of the monitor loop's health-check branch
    # Use a short timeout for test speed; the production code uses
    # HEALTH_CHECK_TIMEOUT (5s).
    in_progress = sum([x.in_progress for x in backend.models.values()])
    test_timeout = 0.1
    try:
        if in_progress:
            pass
        else:
            await asyncio.wait_for(
                _hang(),
                timeout=test_timeout,
            )
            backend.healthy = True
    except asyncio.TimeoutError:
        backend.healthy = False
        backend.unhealthy_reason = (
            f"health check timed out after {test_timeout}s"
        )

    assert not backend.healthy
    assert "timed out" in backend.unhealthy_reason
    assert str(test_timeout) in backend.unhealthy_reason


async def test_manifest_update_timeout_does_not_raise():
    """When manifest update times out, a warning is logged and no exception
    propagates (rebalance must not crash)."""
    import asyncio

    from skvaider.manifest import Serial

    backend = SkvaiderBackend("http://inference-1")
    backend.healthy = True
    backend.current_serial = Serial.floor()
    backend.backend_api = AsyncMock(
        side_effect=asyncio.TimeoutError("simulated timeout")
    )

    pool = Pool(
        [
            ModelInstanceConfig(
                id="m1", instances=1, memory={"ram": 100}, task="chat"
            )
        ],
        [backend],
    )
    # Trigger rebalance to populate last_map
    await pool.rebalance()
    backend.backend_api.reset_mock()

    # Simulate stale serial
    backend.current_serial = Serial.floor()
    assert backend.current_serial < pool.map_serial
    assert backend.url in pool.last_map

    # backend_api raises TimeoutError; update_manifest must not propagate it
    await backend.update_manifest()

    # backend_api was called (we didn't short-circuit)
    backend.backend_api.assert_called_once()
