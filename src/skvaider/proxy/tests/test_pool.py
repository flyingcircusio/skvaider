from skvaider.config import ModelInstanceConfig, parse_size
from skvaider.utils import TaskManager

from ..backends import DummyBackend
from ..models import AIModel
from ..pool import Pool


async def test_maps_only_includes_desired_models(
    task_managers: list[TaskManager],
):

    backend = DummyBackend("http://example.com/")
    backend.healthy = True

    model1 = AIModel(id="m1", owned_by="fcio", backend=backend)
    model1.memory_usage = {"ram": 1}
    model2 = AIModel(id="m2", owned_by="fcio", backend=backend)
    model2.memory_usage = {"ram": 1}

    backend.memory = {"ram": {"free": 1025, "total": 1024}}
    backend.models["m1"] = model1
    backend.models["m2"] = model2

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
    """rebalance loads model instances to match desired count."""
    backend = DummyBackend("http://example.com/")
    backend.healthy = True
    backend.memory = {"ram": {"free": 1024, "total": 1024}}

    model1 = AIModel(id="m1", owned_by="fcio", backend=backend)
    model1.memory_usage = {"ram": 100}
    backend.models["m1"] = model1

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
    """rebalance loads instances across multiple backends."""

    backend1 = DummyBackend("http://example.com/1")
    backend1.healthy = True
    backend1.memory = {"ram": {"free": 500, "total": 500}}

    backend2 = DummyBackend("http://example.com/2")
    backend2.healthy = True
    backend2.memory = {"ram": {"free": 500, "total": 500}}

    model1_b1 = AIModel(id="m1", owned_by="fcio", backend=backend1)
    model1_b1.memory_usage = {"ram": 100}
    backend1.models["m1"] = model1_b1

    model1_b2 = AIModel(id="m1", owned_by="fcio", backend=backend2)
    model1_b2.memory_usage = {"ram": 100}
    backend2.models["m1"] = model1_b2

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
    """rebalance unloads instances when there are more than desired."""
    backend1 = DummyBackend("http://example.com/1")
    backend1.healthy = True
    backend1.memory = {
        "ram": {"free": parse_size("500K"), "total": parse_size("500K")}
    }

    backend2 = DummyBackend("http://example.com/2")
    backend2.healthy = True
    backend2.memory = {
        "ram": {"free": parse_size("500K"), "total": parse_size("500K")}
    }

    model1_b1 = AIModel(id="m1", owned_by="fcio", backend=backend1)
    model1_b1.memory_usage = {"ram": parse_size("100K")}
    backend1.models["m1"] = model1_b1

    model1_b2 = AIModel(id="m1", owned_by="fcio", backend=backend2)
    model1_b2.memory_usage = {"ram": parse_size("100K")}
    backend2.models["m1"] = model1_b2

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


async def test_rebalance_respects_capacity(task_managers: list[TaskManager]):
    backend = DummyBackend("http://example.com/")
    backend.healthy = True
    backend.memory = {
        "ram": {"free": parse_size("150K"), "total": parse_size("150K")}
    }

    model1 = AIModel(id="m1", owned_by="fcio", backend=backend)
    model1.memory_usage = {"ram": parse_size("100K")}
    backend.models["m1"] = model1

    model2 = AIModel(id="m2", owned_by="fcio", backend=backend)
    model2.memory_usage = {"ram": parse_size("100K")}
    backend.models["m2"] = model2

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
    """rebalance ignores unhealthy backends."""
    backend1 = DummyBackend("http://example.com/1")
    backend1.healthy = True
    backend1.memory = {
        "ram": {"free": parse_size("500K"), "total": parse_size("500K")}
    }

    backend2 = DummyBackend("http://example.com/2")
    backend2.healthy = False
    backend2.memory = {
        "ram": {"free": parse_size("500K"), "total": parse_size("500K")}
    }

    model1_b1 = AIModel(id="m1", owned_by="fcio", backend=backend1)
    model1_b1.memory_usage = {"ram": parse_size("100K")}
    backend1.models["m1"] = model1_b1

    model1_b2 = AIModel(id="m1", owned_by="fcio", backend=backend2)
    model1_b2.memory_usage = {"ram": parse_size("100K")}
    backend2.models["m1"] = model1_b2

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
    backend1 = DummyBackend("http://example.com/1")
    backend1.healthy = True
    backend1.memory = {
        "ram": {"free": parse_size("200K"), "total": parse_size("200K")}
    }

    backend2 = DummyBackend("http://example.com/2")
    backend2.healthy = False
    backend2.memory = {
        "ram": {"free": parse_size("200K"), "total": parse_size("200K")}
    }

    model1_b1 = AIModel(id="m1", owned_by="fcio", backend=backend1)
    model1_b1.memory_usage = {"ram": parse_size("100K")}
    backend1.models["m1"] = model1_b1

    model1_b2 = AIModel(id="m1", owned_by="fcio", backend=backend2)
    model1_b2.memory_usage = {"ram": parse_size("100K")}
    backend2.models["m1"] = model1_b2

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
    """When backend becomes unhealthy, its models don't count."""

    backend1 = DummyBackend("http://example.com/1")
    backend1.healthy = True
    backend1.memory = {
        "ram": {"free": parse_size("200K"), "total": parse_size("200K")}
    }

    backend2 = DummyBackend("http://example.com/2")
    backend2.healthy = True
    backend2.memory = {
        "ram": {"free": parse_size("200K"), "total": parse_size("200K")}
    }

    model1_b1 = AIModel(id="m1", owned_by="fcio", backend=backend1)
    model1_b1.memory_usage = {"ram": parse_size("100K")}
    backend1.models["m1"] = model1_b1

    model1_b2 = AIModel(id="m1", owned_by="fcio", backend=backend2)
    model1_b2.memory_usage = {"ram": parse_size("100K")}
    backend2.models["m1"] = model1_b2

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
    """In complex scenarios we want to see that models get moved around."""
    backend1 = DummyBackend("http://example.com/1")
    backend1.healthy = True
    backend1.memory = {
        "ram": {"free": parse_size("400K"), "total": parse_size("400K")}
    }

    backend2 = DummyBackend("http://example.com/2")
    backend2.healthy = True
    backend2.memory = {
        "ram": {"free": parse_size("400K"), "total": parse_size("400K")}
    }

    backend3 = DummyBackend("http://example.com/3")
    backend3.healthy = True
    backend3.memory = {
        "ram": {"free": parse_size("400K"), "total": parse_size("400K")}
    }

    model1_b1 = AIModel(id="m1", owned_by="fcio", backend=backend1)
    backend1.models["m1"] = model1_b1

    model1_b2 = AIModel(id="m1", owned_by="fcio", backend=backend2)
    backend2.models["m1"] = model1_b2

    model1_b3 = AIModel(id="m1", owned_by="fcio", backend=backend3)
    backend3.models["m1"] = model1_b3

    model2_b1 = AIModel(id="m2", owned_by="fcio", backend=backend1)
    backend1.models["m2"] = model2_b1

    model2_b2 = AIModel(id="m2", owned_by="fcio", backend=backend2)
    backend2.models["m2"] = model2_b2

    model2_b3 = AIModel(id="m2", owned_by="fcio", backend=backend3)
    backend3.models["m2"] = model2_b3

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
