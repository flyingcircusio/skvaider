import pytest

from skvaider.config import ModelInstanceConfig, parse_size
from skvaider.conftest import registered_model_factory
from skvaider.proxy.backends import DummyBackend
from skvaider.utils import TaskManager

from ..pool import Pool


async def test_configured_memory_returns_empty_when_not_in_pool_config(
    task_managers: list[TaskManager],
    dummy_backend: DummyBackend,
):
    pool = Pool([], [dummy_backend])
    task_managers.append(pool.tasks)
    model = registered_model_factory("x", dummy_backend)
    assert model.configured_memory == {}


async def test_configured_memory_returns_pool_config_memory(
    task_managers: list[TaskManager],
    dummy_backend: DummyBackend,
):
    model = registered_model_factory("m", dummy_backend)
    pool = Pool(
        [
            ModelInstanceConfig(
                id="m",
                instances=1,
                memory={"ram": parse_size("100K")},
                task="chat",
            )
        ],
        [dummy_backend],
    )
    task_managers.append(pool.tasks)
    assert model.configured_memory == {"ram": 100 * 1024}


async def test_total_size_sums_configured_memory(
    task_managers: list[TaskManager],
    dummy_backend: DummyBackend,
):
    model = registered_model_factory("m", dummy_backend)
    pool = Pool(
        [
            ModelInstanceConfig(
                id="m",
                instances=1,
                memory={"ram": parse_size("1K"), "vram": parse_size("2K")},
                task="chat",
            )
        ],
        [dummy_backend],
    )
    task_managers.append(pool.tasks)
    assert model.total_size() == 3 * 1024


async def test_total_size_zero_when_no_config(
    task_managers: list[TaskManager],
    dummy_backend: DummyBackend,
):
    pool = Pool([], [dummy_backend])
    task_managers.append(pool.tasks)
    model = registered_model_factory("x", dummy_backend)
    assert model.total_size() == 0


async def test_check_memory_usage_returns_exceeding_resources(
    task_managers: list[TaskManager],
    dummy_backend: DummyBackend,
):
    model = registered_model_factory("m", dummy_backend, ram=200)
    pool = Pool(
        [
            ModelInstanceConfig(
                id="m", instances=1, memory={"ram": 100}, task="chat"
            )
        ],
        [dummy_backend],
    )
    task_managers.append(pool.tasks)
    assert model.config.check_memory_usage(model.memory_usage) == {
        "ram": (200, 100)
    }


async def test_check_memory_usage_empty_when_within_limits(
    task_managers: list[TaskManager],
    dummy_backend: DummyBackend,
):
    model = registered_model_factory("m", dummy_backend, ram=50)
    pool = Pool(
        [
            ModelInstanceConfig(
                id="m", instances=1, memory={"ram": 100}, task="chat"
            )
        ],
        [dummy_backend],
    )
    task_managers.append(pool.tasks)
    assert model.config.check_memory_usage(model.memory_usage) == {}


async def test_fit_score_uses_backend_free_memory(
    task_managers: list[TaskManager],
    dummy_backend: DummyBackend,
):
    # available=1000, usage=200 → score = 1 - 200/1000 = 0.8
    model = registered_model_factory("m", dummy_backend)
    pool = Pool(
        [
            ModelInstanceConfig(
                id="m", instances=1, memory={"ram": 200}, task="chat"
            )
        ],
        [dummy_backend],
    )
    task_managers.append(pool.tasks)
    assert model.fit_score() == pytest.approx(0.8)  # type: ignore


async def test_fit_score_adds_usage_back_when_already_loaded(
    task_managers: list[TaskManager],
    dummy_backend: DummyBackend,
):
    # loaded: available = free(1000) + usage(200) = 1200 → score = 1 - 200/1200
    model = registered_model_factory("m", dummy_backend, loaded=True)
    pool = Pool(
        [
            ModelInstanceConfig(
                id="m", instances=1, memory={"ram": 200}, task="chat"
            )
        ],
        [dummy_backend],
    )
    task_managers.append(pool.tasks)
    assert model.fit_score() == pytest.approx(  # pyright: ignore[reportUnknownMemberType]
        1 - 200 / 1200
    )


async def test_fit_score_returns_zero_when_does_not_fit(
    task_managers: list[TaskManager],
    dummy_backend: DummyBackend,
):
    # available(100) < usage(200) → resource skipped → score stays 0.0
    dummy_backend.memory = {"ram": {"free": 100, "total": 100}}
    model = registered_model_factory("m", dummy_backend)
    pool = Pool(
        [
            ModelInstanceConfig(
                id="m", instances=1, memory={"ram": 200}, task="chat"
            )
        ],
        [dummy_backend],
    )
    task_managers.append(pool.tasks)
    assert model.fit_score() == 0.0


async def test_fit_score_with_explicit_resources_dict(
    task_managers: list[TaskManager],
    dummy_backend: DummyBackend,
):
    # explicit resources override backend free memory: 1 - 200/500 = 0.6
    model = registered_model_factory("m", dummy_backend)
    pool = Pool(
        [
            ModelInstanceConfig(
                id="m", instances=1, memory={"ram": 200}, task="chat"
            )
        ],
        [dummy_backend],
    )
    task_managers.append(pool.tasks)
    assert model.fit_score(resources={"ram": 500}) == pytest.approx(  # pyright: ignore[reportUnknownMemberType]
        0.6
    )
