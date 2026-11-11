from unittest.mock import AsyncMock

from skvaider.config import ModelInstanceConfig
from skvaider.manifest import MaintenanceStatus, ManifestRequest
from skvaider.proxy.backends import SkvaiderBackend
from skvaider.proxy.models import AIModel
from skvaider.proxy.pool import Pool
from skvaider.utils import TaskManager


class RecordingBackend(SkvaiderBackend):
    """Skvaider backend without the health monitor, driven by the test."""

    api: AsyncMock

    def __init__(self, url: str):
        super().__init__(url)
        self.api = AsyncMock()
        self.backend_api = self.api

    async def monitor_health_and_update_models(self) -> None:
        pass


def make_backend(url: str, ram: int = 1024) -> RecordingBackend:
    backend = RecordingBackend(url)
    backend.healthy = True
    backend.map_up.mark("up")
    backend.memory = {"ram": {"free": ram, "total": ram}}
    return backend


def add_model(
    backend: RecordingBackend, model_id: str, loaded: bool
) -> AIModel:
    model = AIModel(id=model_id, owned_by="test", backend=backend)
    model.memory_usage = {"ram": 100}
    model.is_loaded = loaded
    backend.models[model_id] = model
    return model


def make_pool(
    backends: list[RecordingBackend],
    task_managers: list[TaskManager],
    instances: int = 1,
) -> Pool:
    pool = Pool(
        [
            ModelInstanceConfig(
                id="m1", instances=instances, memory={"ram": 100}, task="chat"
            )
        ],
        backends,
    )
    task_managers.append(pool.tasks)
    return pool


def manifest_requests(backend: RecordingBackend) -> list[set[str]]:
    requests: list[set[str]] = []
    for call in backend.api.call_args_list:
        if call.args and isinstance(call.args[0], ManifestRequest):
            requests.append(set(call.args[0].models))
    return requests


async def test_maintenance_takes_backend_out_of_map_and_waits_for_replacement(
    task_managers: list[TaskManager],
):
    backend1 = make_backend("http://backend-1")
    backend2 = make_backend("http://backend-2")
    add_model(backend1, "m1", loaded=True)
    replacement = add_model(backend2, "m1", loaded=False)

    pool = make_pool([backend1, backend2], task_managers)
    pool.last_map = {backend1.url: {"m1"}, backend2.url: set()}

    await backend1.observe_maintenance("requested")

    assert backend1.map_in == "out"
    assert pool.last_map == {backend1.url: set(), backend2.url: {"m1"}}
    # The replacement was asked to load...
    assert manifest_requests(backend2) == [{"m1"}]
    # ...but the old backend is not asked to unload before that happened.
    assert manifest_requests(backend1) == []

    # The replacement finished loading and the old backend gets emptied.
    replacement.is_loaded = True
    await backend1.update_manifest()
    assert manifest_requests(backend1) == [set()]


async def test_maintenance_blocked_without_room_elsewhere(
    task_managers: list[TaskManager],
):
    backend = make_backend("http://backend-1")
    model = add_model(backend, "m1", loaded=True)

    pool = make_pool([backend], task_managers)
    pool.last_map = {backend.url: {"m1"}}

    await backend.observe_maintenance("requested")

    assert backend.map_in == "in"
    assert backend.maintenance_blocked
    assert pool.last_map == {backend.url: {"m1"}}
    assert model.is_loaded
    assert manifest_requests(backend) == []


async def test_maintenance_is_polled_while_requests_are_in_flight(
    task_managers: list[TaskManager],
):
    backend1 = make_backend("http://backend-1")
    backend2 = make_backend("http://backend-2")
    model = add_model(backend1, "m1", loaded=True)
    model.in_progress = 3
    add_model(backend2, "m1", loaded=False)

    pool = make_pool([backend1, backend2], task_managers)
    pool.last_map = {backend1.url: {"m1"}, backend2.url: set()}

    backend1.api = AsyncMock(
        return_value=MaintenanceStatus(
            phase="requested", manifest=["m1"], pending_models=[]
        )
    )
    backend1.backend_api = backend1.api

    await backend1._monitor_health_and_update_models()  # pyright: ignore[reportPrivateUsage]

    # The health check was skipped, but the maintenance request was seen.
    assert backend1.map_in == "out"
    requests = [call.args[0] for call in backend1.api.call_args_list]
    assert [type(request).__name__ for request in requests] == [
        "MaintenanceStatusRequest"
    ]


async def test_leaving_maintenance_puts_backend_back_in(
    task_managers: list[TaskManager],
):
    backend = make_backend("http://backend-1")
    add_model(backend, "m1", loaded=False)

    pool = make_pool([backend], task_managers)
    pool.last_map = {backend.url: set()}

    await backend.observe_maintenance("requested")
    assert backend.map_in == "out"

    await backend.observe_maintenance("none")
    assert backend.map_in == "in"
    assert backend.maintenance == "none"
