import asyncio

from skvaider import utils

from ..backends import DummyBackend
from ..models import AIModel
from ..pool import Pool


async def wait_for_cancelled_tasks():
    tasks = [
        t
        for t in asyncio.all_tasks()
        if t is not asyncio.current_task() and t.cancelling()
    ]

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_update_model_maps():
    pool = Pool()
    assert pool.tasks.count == 0

    backend = DummyBackend("http://example.com/", pool)
    pool.add_backend(backend)
    assert pool.tasks.count == 1

    model1 = AIModel(id="m1", owned_by="fcio", backend=backend)
    model1.memory_usage = {"cpu": 1}
    assert not model1.is_loaded

    model2 = AIModel(id="m2", owned_by="fcio", backend=backend)
    model2.memory_usage = {"cpu": 1}
    assert not model2.is_loaded

    backend.memory = {"cpu": {"free": 1024, "total": 1024}}

    assert not pool.models
    pool.update_model_maps()
    assert not pool.models

    # Add a model
    backend.models["m1"] = model1
    pool.update_model_maps()
    assert pool.tasks.count == 3
    await pool.tasks.unique_task_map["m1:add_model_instance"]
    assert pool.tasks.count == 2
    assert pool.tasks.unique_task_map.keys() == {
        "m1:queue",
    }
    assert pool.models == {"m1"}
    assert model1.is_loaded

    # Add another model
    backend.models["m2"] = model2
    pool.update_model_maps()
    assert pool.tasks.count == 4
    await pool.tasks.unique_task_map["m2:add_model_instance"]
    assert pool.tasks.count == 3
    assert pool.tasks.unique_task_map.keys() == {
        "m1:queue",
        "m2:queue",
    }
    assert pool.models == {"m1", "m2"}
    assert model2.is_loaded

    # Remove a model
    del backend.models["m2"]
    pool.update_model_maps()
    await wait_for_cancelled_tasks()
    assert pool.tasks.count == 2
    assert pool.tasks.unique_task_map.keys() == {
        "m1:queue",
    }
    assert pool.models == {"m1"}

    # Remove last model
    del backend.models["m1"]
    pool.update_model_maps()
    await wait_for_cancelled_tasks()
    assert pool.tasks.count == 1
    assert not pool.tasks.unique_task_map
    assert not pool.models


async def test_add_model_instance_gives_up():
    pool = Pool()
    backend = DummyBackend("http://example.com/", pool)
    pool.add_backend(backend)
    assert pool.tasks.count == 1

    model1 = AIModel(id="m1", owned_by="fcio", backend=backend)
    model1.memory_usage = {"cpu": 1}
    assert not model1.is_loaded

    # The model won't fit.
    backend.memory = {"cpu": {"free": 0, "total": 0}}

    backend.models["m1"] = model1
    pool.update_model_maps()
    await pool.tasks.unique_task_map["m1:add_model_instance"]
    assert pool.tasks.count == 2
    assert pool.tasks.unique_task_map.keys() == {
        "m1:queue",
    }
    assert not model1.is_loaded


async def test_make_room():
    pool = Pool()

    backends: dict[int, DummyBackend] = {}
    models: dict[tuple[int, int], AIModel] = dict()

    # First backend will load all models
    backends[0] = b = DummyBackend("http://example.com/0", pool)
    pool.add_backend(b)
    b.memory = {"ram": {"free": 1001, "total": 1001}}

    for i_m in range(5):
        models[(0, i_m)] = m = AIModel(
            id=f"m_{i_m}", owned_by="fcio", backend=b
        )
        m.memory_usage = {"ram": 250}

        if i_m < 4:
            b.models[m.id] = m
            pool.update_model_maps()
            await pool.tasks.unique_task_map[f"{m.id}:add_model_instance"]
            assert m.is_loaded

    # We can fit 4 models in the pool and then we have to make room for the 5th. Note: we never
    # allocate the last byte (to keep the math in fit_score() simpler).

    # Second backend is empty for now. Lets also load all 4 first models there
    backends[1] = b = DummyBackend("http://example.com/1", pool)
    b.memory = {"ram": {"free": 1001, "total": 1001}}
    pool.add_backend(b)
    for i_m in range(5):
        models[(1, i_m)] = m = AIModel(
            id=f"m_{i_m}", owned_by="fcio", backend=b
        )
        m.memory_usage = {"ram": 250}

        if i_m < 4:
            b.models[m.id] = m
            pool.update_model_maps()
            await pool.add_model_instance(m.id)
            assert m.is_loaded

    assert backends[0].memory == {"ram": {"free": 1, "total": 1001}}
    assert backends[1].memory == {"ram": {"free": 1, "total": 1001}}

    loaded_models_1 = set([m.id for m in models.values() if m.is_loaded])

    backends[0].models["m_4"] = models[(0, 4)]
    pool.update_model_maps()
    # This will trigger trying to load model 8, but that won't happen
    # as the other models have only just been loaded.
    await pool.tasks.unique_task_map["m_4:add_model_instance"]
    loaded_models_2 = set([m.id for m in models.values() if m.is_loaded])
    assert loaded_models_1 == loaded_models_2
    assert not models[(0, 4)].is_loaded

    # Now, set the last used flag of an already loaded model on backend 0
    # and then explicitly try to load model 4 again.
    loaded = [
        m for m in models.values() if m.is_loaded and m.backend is backends[0]
    ][0]
    loaded.last_used = utils.datetime_min
    not_loaded = models[(0, 4)]
    pool.ensure_reserved_instance(not_loaded.id)
    await pool.tasks.unique_task_map["m_4:add_model_instance"]
    assert not loaded.is_loaded
    assert not_loaded.is_loaded

    assert backends[0].memory == {"ram": {"free": 1, "total": 1001}}
    assert backends[1].memory == {"ram": {"free": 1, "total": 1001}}
