from skvaider.config import ModelInstanceConfig, parse_size
from skvaider.conftest import backend_factory, registered_model_factory
from skvaider.utils import TaskManager

from ..pool import Pool


async def test_maps_only_includes_desired_models(
    task_managers: list[TaskManager],
):
    backend = backend_factory("http://example.com/", ram=1024)
    backend.memory = {"ram": {"free": 1025, "total": 1024}}
    registered_model_factory("m1", backend, ram=1)
    registered_model_factory("m2", backend, ram=1)

    pool = Pool(
        [ModelInstanceConfig(id="m1", instances=1, memory={"ram": 1})],
        [backend],
    )
    task_managers.append(pool.tasks)
    await pool.rebalance()

    assert pool.model_configs.keys() == {"m1"}
    assert "m1" in pool.semaphores
    assert "m2" not in pool.semaphores


async def test_rebalance_loads_desired_instances(
    task_managers: list[TaskManager],
):
    backend = backend_factory("http://example.com/", ram=1024)
    model1 = registered_model_factory("m1", backend, ram=100)

    assert not model1.is_loaded

    pool = Pool(
        [ModelInstanceConfig(id="m1", instances=1, memory={"ram": 100})],
        [backend],
    )
    task_managers.append(pool.tasks)

    await pool.rebalance()
    assert model1.is_loaded


async def test_rebalance_distributes_across_backends(
    task_managers: list[TaskManager],
):
    backend1 = backend_factory("http://example.com/1", ram=500)
    backend2 = backend_factory("http://example.com/2", ram=500)
    model1_b1 = registered_model_factory("m1", backend1, ram=100)
    model1_b2 = registered_model_factory("m1", backend2, ram=100)

    pool = Pool(
        [ModelInstanceConfig(id="m1", instances=2, memory={"ram": 100})],
        [backend1, backend2],
    )
    task_managers.append(pool.tasks)

    await pool.rebalance()

    assert model1_b1.is_loaded
    assert model1_b2.is_loaded
    assert pool.count_loaded_instances("m1") == 2


async def test_rebalance_unloads_excess_instances(
    task_managers: list[TaskManager],
):
    backend1 = backend_factory("http://example.com/1", ram=parse_size("500K"))
    backend2 = backend_factory("http://example.com/2", ram=parse_size("500K"))
    registered_model_factory("m1", backend1, ram=parse_size("100K"))
    registered_model_factory("m1", backend2, ram=parse_size("100K"))

    model1_config = ModelInstanceConfig(
        id="m1", instances=2, memory={"ram": parse_size("100K")}
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
):
    backend = backend_factory("http://example.com/", ram=parse_size("150K"))
    registered_model_factory("m1", backend, ram=parse_size("100K"))
    registered_model_factory("m2", backend, ram=parse_size("100K"))

    model1_config = ModelInstanceConfig(
        id="m1", instances=1, memory={"ram": parse_size("100K")}
    )
    model2_config = ModelInstanceConfig(
        id="m2", instances=1, memory={"ram": parse_size("100K")}
    )
    pool = Pool([model1_config, model2_config], [backend])
    task_managers.append(pool.tasks)
    await pool.rebalance()

    total_loaded = pool.count_loaded_instances(
        "m1"
    ) + pool.count_loaded_instances("m2")
    assert total_loaded == 1


async def test_rebalance_handles_unhealthy_backend(
    task_managers: list[TaskManager],
):
    backend1 = backend_factory("http://example.com/1", ram=parse_size("500K"))
    backend2 = backend_factory("http://example.com/2", ram=parse_size("500K"))
    backend2.healthy = False
    model1_b1 = registered_model_factory("m1", backend1, ram=parse_size("100K"))
    model1_b2 = registered_model_factory("m1", backend2, ram=parse_size("100K"))

    model1_config = ModelInstanceConfig(
        id="m1", instances=2, memory={"ram": parse_size("100K")}
    )
    pool = Pool([model1_config], [backend1, backend2])
    task_managers.append(pool.tasks)

    await pool.rebalance()
    assert model1_b1.is_loaded

    assert not model1_b2.is_loaded
    assert pool.count_loaded_instances("m1") == 1


async def test_rebalance_after_backend_becomes_healthy(
    task_managers: list[TaskManager],
):
    backend1 = backend_factory("http://example.com/1", ram=parse_size("200K"))
    backend2 = backend_factory("http://example.com/2", ram=parse_size("200K"))
    backend2.healthy = False
    model1_b1 = registered_model_factory("m1", backend1, ram=parse_size("100K"))
    model1_b2 = registered_model_factory("m1", backend2, ram=parse_size("100K"))

    model1_config = ModelInstanceConfig(
        id="m1", instances=2, memory={"ram": parse_size("100K")}
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
):
    backend1 = backend_factory("http://example.com/1", ram=parse_size("200K"))
    backend2 = backend_factory("http://example.com/2", ram=parse_size("200K"))
    registered_model_factory("m1", backend1, ram=parse_size("100K"))
    registered_model_factory("m1", backend2, ram=parse_size("100K"))

    model1_config = ModelInstanceConfig(
        id="m1", instances=2, memory={"ram": parse_size("100K")}
    )
    pool = Pool([model1_config], [backend1, backend2])
    task_managers.append(pool.tasks)

    await pool.rebalance()
    assert pool.count_loaded_instances("m1") == 2

    backend2.healthy = False

    await pool.rebalance()
    assert pool.count_loaded_instances("m1") == 1


async def test_complex_rebalance_multiple_models(
    task_managers: list[TaskManager],
):
    """Models get moved around when backends change health."""
    backend1 = backend_factory("http://example.com/1", ram=parse_size("400K"))
    backend2 = backend_factory("http://example.com/2", ram=parse_size("400K"))
    backend3 = backend_factory("http://example.com/3", ram=parse_size("400K"))

    registered_model_factory("m1", backend1)
    registered_model_factory("m1", backend2)
    registered_model_factory("m1", backend3)
    registered_model_factory("m2", backend1)
    registered_model_factory("m2", backend2)
    registered_model_factory("m2", backend3)

    model1_config = ModelInstanceConfig(
        id="m1", instances=2, memory={"ram": parse_size("100K")}
    )
    model2_config = ModelInstanceConfig(
        id="m2", instances=2, memory={"ram": parse_size("200K")}
    )
    pool = Pool(
        [model1_config, model2_config],
        [backend1, backend2, backend3],
    )
    task_managers.append(pool.tasks)

    await pool.rebalance()
    assert pool.count_loaded_instances("m1") == 2
    assert pool.count_loaded_instances("m2") == 2

    backend2.healthy = False
    await pool.rebalance()

    assert pool.count_loaded_instances("m1") == 2
    assert pool.count_loaded_instances("m2") == 2
