import asyncio
import datetime
from typing import Callable
from unittest.mock import patch

from skvaider.config import ModelInstanceConfig, parse_size
from skvaider.conftest import registered_model_factory
from skvaider.proxy.backends import DummyBackend
from skvaider.proxy.models import AIModel
from skvaider.utils import TaskManager

from ..pool import Pool


async def test_maps_only_includes_desired_models(
    task_managers: list[TaskManager],
    dummy_backend: DummyBackend,
):
    dummy_backend.memory = {"ram": {"free": 1025, "total": 1024}}
    registered_model_factory("m1", dummy_backend, ram=1)
    registered_model_factory("m2", dummy_backend, ram=1)

    pool = Pool(
        [
            ModelInstanceConfig(
                id="m1", instances=1, memory={"ram": 1}, task="chat"
            )
        ],
        [dummy_backend],
    )
    task_managers.append(pool.tasks)
    await pool.rebalance()

    assert pool.model_configs.keys() == {"m1"}
    assert "m1" in pool.semaphores
    assert "m2" not in pool.semaphores


async def test_rebalance_loads_desired_instances(
    task_managers: list[TaskManager],
    dummy_backend: DummyBackend,
):
    dummy_backend.memory = {"ram": {"free": 1024, "total": 1024}}
    model1 = registered_model_factory("m1", dummy_backend, ram=100)
    assert not model1.is_loaded

    pool = Pool(
        [
            ModelInstanceConfig(
                id="m1", instances=1, memory={"ram": 100}, task="chat"
            )
        ],
        [dummy_backend],
    )
    task_managers.append(pool.tasks)

    await pool.rebalance()
    assert model1.is_loaded


async def test_rebalance_distributes_across_backends(
    task_managers: list[TaskManager],
    dummy_backend_factory: Callable[..., DummyBackend],
):
    backend1 = dummy_backend_factory(ram=500)
    backend2 = dummy_backend_factory(ram=500)
    model1_b1 = registered_model_factory("m1", backend1, ram=100)
    model1_b2 = registered_model_factory("m1", backend2, ram=100)

    pool = Pool(
        [
            ModelInstanceConfig(
                id="m1", instances=2, memory={"ram": 100}, task="chat"
            )
        ],
        [backend1, backend2],
    )
    task_managers.append(pool.tasks)

    await pool.rebalance()

    assert model1_b1.is_loaded
    assert model1_b2.is_loaded
    assert pool.count_loaded_instances("m1") == 2


async def test_rebalance_unloads_excess_instances(
    task_managers: list[TaskManager],
    dummy_backend_factory: Callable[..., DummyBackend],
):
    backend1 = dummy_backend_factory(ram=parse_size("500K"))
    backend2 = dummy_backend_factory(ram=parse_size("500K"))
    registered_model_factory("m1", backend1, ram=parse_size("100K"))
    registered_model_factory("m1", backend2, ram=parse_size("100K"))

    model1_config = ModelInstanceConfig(
        id="m1", instances=2, memory={"ram": parse_size("100K")}, task="chat"
    )

    pool = Pool([model1_config], [backend1, backend2])
    task_managers.append(pool.tasks)

    await pool.rebalance()
    assert pool.count_loaded_instances("m1") == 2

    pool.model_configs["m1"].instances = 1
    await pool.rebalance()
    assert pool.count_loaded_instances("m1") == 1


async def test_rebalance_respects_capacity(
    task_managers: list[TaskManager],
    dummy_backend: DummyBackend,
):
    dummy_backend.memory = {
        "ram": {"free": parse_size("150K"), "total": parse_size("150K")}
    }
    registered_model_factory("m1", dummy_backend, ram=parse_size("100K"))
    registered_model_factory("m2", dummy_backend, ram=parse_size("100K"))

    model1_config = ModelInstanceConfig(
        id="m1", instances=1, memory={"ram": parse_size("100K")}, task="chat"
    )
    model2_config = ModelInstanceConfig(
        id="m2", instances=1, memory={"ram": parse_size("100K")}, task="chat"
    )
    pool = Pool([model1_config, model2_config], [dummy_backend])
    task_managers.append(pool.tasks)
    await pool.rebalance()

    total_loaded = pool.count_loaded_instances(
        "m1"
    ) + pool.count_loaded_instances("m2")
    assert total_loaded == 1


async def test_rebalance_handles_unhealthy_backend(
    task_managers: list[TaskManager],
    dummy_backend_factory: Callable[..., DummyBackend],
):
    backend1 = dummy_backend_factory(ram=parse_size("500K"))
    backend2 = dummy_backend_factory(ram=parse_size("500K"))
    backend2.healthy = False
    model1_b1 = registered_model_factory("m1", backend1, ram=parse_size("100K"))
    model1_b2 = registered_model_factory("m1", backend2, ram=parse_size("100K"))

    model1_config = ModelInstanceConfig(
        id="m1", instances=2, memory={"ram": parse_size("100K")}, task="chat"
    )
    pool = Pool([model1_config], [backend1, backend2])
    task_managers.append(pool.tasks)

    await pool.rebalance()
    assert model1_b1.is_loaded

    assert not model1_b2.is_loaded
    assert pool.count_loaded_instances("m1") == 1


async def test_rebalance_after_backend_becomes_healthy(
    task_managers: list[TaskManager],
    dummy_backend_factory: Callable[..., DummyBackend],
):
    backend1 = dummy_backend_factory(ram=parse_size("200K"))
    backend2 = dummy_backend_factory(ram=parse_size("200K"))
    backend2.healthy = False
    model1_b1 = registered_model_factory("m1", backend1, ram=parse_size("100K"))
    model1_b2 = registered_model_factory("m1", backend2, ram=parse_size("100K"))

    model1_config = ModelInstanceConfig(
        id="m1", instances=2, memory={"ram": parse_size("100K")}, task="chat"
    )
    pool = Pool([model1_config], [backend1, backend2])
    task_managers.append(pool.tasks)

    await pool.rebalance()
    assert pool.count_loaded_instances("m1") == 1

    backend2.healthy = True
    await pool.rebalance()
    assert pool.count_loaded_instances("m1") == 2
    assert model1_b1.is_loaded
    assert model1_b2.is_loaded


async def test_rebalance_after_backend_becomes_unhealthy(
    task_managers: list[TaskManager],
    dummy_backend_factory: Callable[..., DummyBackend],
):
    backend1 = dummy_backend_factory(ram=parse_size("200K"))
    backend2 = dummy_backend_factory(ram=parse_size("200K"))
    registered_model_factory("m1", backend1, ram=parse_size("100K"))
    registered_model_factory("m1", backend2, ram=parse_size("100K"))

    model1_config = ModelInstanceConfig(
        id="m1", instances=2, memory={"ram": parse_size("100K")}, task="chat"
    )
    pool = Pool([model1_config], [backend1, backend2])
    task_managers.append(pool.tasks)

    await pool.rebalance()
    assert pool.count_loaded_instances("m1") == 2
    assert pool.placement_map() == {
        "http://backend-1": {"m1"},
        "http://backend-2": {"m1"},
    }

    backend2.healthy = False

    await pool.rebalance()

    # This still says 2 as we do not issue unload requests
    # on unhealthy backends to avoid noise.
    assert pool.count_loaded_instances("m1") == 2
    assert pool.placement_map() == {
        "http://backend-1": {"m1"},
        "http://backend-2": set(),
    }


async def test_complex_rebalance_multiple_models(
    task_managers: list[TaskManager],
    dummy_backend_factory: Callable[..., DummyBackend],
):
    """Models get moved around when backends change health."""
    backend1 = dummy_backend_factory(ram=parse_size("400K"))
    backend2 = dummy_backend_factory(ram=parse_size("400K"))
    backend3 = dummy_backend_factory(ram=parse_size("400K"))

    registered_model_factory("m1", backend1)
    registered_model_factory("m1", backend2)
    registered_model_factory("m1", backend3)
    registered_model_factory("m2", backend1)
    registered_model_factory("m2", backend2)
    registered_model_factory("m2", backend3)

    model1_config = ModelInstanceConfig(
        id="m1", instances=2, memory={"ram": parse_size("100K")}, task="chat"
    )
    model2_config = ModelInstanceConfig(
        id="m2", instances=2, memory={"ram": parse_size("200K")}, task="chat"
    )
    pool = Pool(
        [model1_config, model2_config],
        [backend1, backend2, backend3],
    )
    task_managers.append(pool.tasks)

    await pool.rebalance()
    assert pool.count_loaded_instances("m1") == 2
    assert pool.count_loaded_instances("m2") == 2
    assert pool.placement_map() == {
        "http://backend-1": {"m1"},
        "http://backend-2": {"m2"},
        "http://backend-3": {"m2", "m1"},
    }

    backend2.healthy = False
    await pool.rebalance()

    assert pool.count_loaded_instances("m1") == 2
    # this says 3 instead of 2 because we do not touch unhealthy
    # backends to avoid superfluous loading/unloading during
    # temporary disruptions.
    assert pool.count_loaded_instances("m2") == 3
    assert pool.placement_map() == {
        "http://backend-1": {"m2", "m1"},
        "http://backend-2": set(),
        "http://backend-3": {"m2", "m1"},
    }


async def test_pool_report_map_contains_backends_and_models(
    task_managers: list[TaskManager],
    dummy_backend: DummyBackend,
):
    registered_model_factory("m", dummy_backend, loaded=True, ram=100)
    pool = Pool(
        [
            ModelInstanceConfig(
                id="m", instances=1, memory={"ram": 100}, task="chat"
            )
        ],
        [dummy_backend],
    )
    task_managers.append(pool.tasks)

    report = pool.report_map()

    assert dummy_backend.url in report
    models_report = report[dummy_backend.url]["models"]
    assert "m" in models_report
    assert models_report["m"]["loaded"] is True
    assert models_report["m"]["memory"] == {"ram": 100}


async def test_pool_close_terminates_task_manager(dummy_backend: DummyBackend):
    pool = Pool([], [dummy_backend])
    # Monitor task was scheduled in __init__ but not yet run
    assert pool.tasks.count == 1
    pool.close()
    assert pool.tasks.count == 0


# test semaphore acquisition used in _execute_with_retry and _execute_stream_with_retry


async def test_semaphore_acquire_returns_loaded_model(
    task_managers: list[TaskManager],
    dummy_backend: DummyBackend,
):
    model = registered_model_factory("m", dummy_backend, loaded=True)
    pool = Pool(
        [
            ModelInstanceConfig(
                id="m", instances=1, memory={"ram": 100}, task="chat"
            )
        ],
        [dummy_backend],
    )
    task_managers.append(pool.tasks)
    sem = pool.semaphores["m"]
    acquired = await sem.acquire()
    assert acquired is model
    await sem.release(model)


async def test_semaphore_acquire_returns_none_when_no_loaded_models(
    task_managers: list[TaskManager],
    dummy_backend: DummyBackend,
):
    # _candidates returns [] when model is not loaded → while never entered → None
    registered_model_factory("m", dummy_backend, loaded=False)
    pool = Pool(
        [
            ModelInstanceConfig(
                id="m", instances=1, memory={"ram": 100}, task="chat"
            )
        ],
        [dummy_backend],
    )
    task_managers.append(pool.tasks)
    sem = pool.semaphores["m"]
    result = await sem.acquire()
    assert result is None


async def test_semaphore_waiter_woken_on_release(  # tests the continue in ModelSemaphore.acquire()
    task_managers: list[TaskManager],
    dummy_backend: DummyBackend,
):
    registered_model_factory("m", dummy_backend, loaded=True, limit=1)
    pool = Pool(
        [
            ModelInstanceConfig(
                id="m", instances=1, memory={"ram": 100}, task="chat"
            )
        ],
        [dummy_backend],
    )
    task_managers.append(pool.tasks)
    sem = pool.semaphores["m"]

    first = await sem.acquire()
    assert first is not None

    waiter_result: list[AIModel | None] = []

    async def waiter() -> None:
        waiter_result.append(await sem.acquire())

    waiter_task = asyncio.create_task(waiter())
    # Give waiter task a chance to start and block on released.wait()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    await sem.release(first)
    await waiter_task

    assert waiter_result[0] is not None
    await sem.release(waiter_result[0])


async def test_semaphore_acquire_returns_none_when_deadline_exceeded(
    task_managers: list[TaskManager],
    dummy_backend: DummyBackend,
):
    base_time = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)
    model = registered_model_factory("m", dummy_backend, loaded=True, limit=1)
    # All capacity occupied — force timeout path
    model.in_progress = 1
    pool = Pool(
        [
            ModelInstanceConfig(
                id="m", instances=1, memory={"ram": 100}, task="chat"
            )
        ],
        [dummy_backend],
    )
    task_managers.append(pool.tasks)
    sem = pool.semaphores["m"]

    side_effects = [
        base_time,
        base_time + datetime.timedelta(seconds=200),
    ]
    with patch("skvaider.proxy.pool.utils.now", side_effect=side_effects):
        result = await sem.acquire()
    assert result is None
    model.in_progress = 0


async def test_semaphore_excluded_backend_not_chosen(
    task_managers: list[TaskManager],
    dummy_backend_factory: Callable[..., DummyBackend],
):
    backend1 = dummy_backend_factory("http://example.com/1")
    backend2 = dummy_backend_factory("http://example.com/2")
    registered_model_factory("m", backend1, loaded=True)
    model2 = registered_model_factory("m", backend2, loaded=True)
    pool = Pool(
        [
            ModelInstanceConfig(
                id="m", instances=2, memory={"ram": 100}, task="chat"
            )
        ],
        [backend1, backend2],
    )
    task_managers.append(pool.tasks)
    sem = pool.semaphores["m"]

    result = await sem.acquire(excluded_backends=[backend1.url])
    assert result is model2
    await sem.release(model2)


async def test_semaphore_use_yields_backend_and_releases(
    task_managers: list[TaskManager],
    dummy_backend: DummyBackend,
):
    model = registered_model_factory("m", dummy_backend, loaded=True)
    pool = Pool(
        [
            ModelInstanceConfig(
                id="m", instances=1, memory={"ram": 100}, task="chat"
            )
        ],
        [dummy_backend],
    )
    task_managers.append(pool.tasks)
    sem = pool.semaphores["m"]

    async with sem.use() as b:
        assert b is dummy_backend
    assert model.in_progress == 0
