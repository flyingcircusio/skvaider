import asyncio
import contextlib
import datetime
from typing import TYPE_CHECKING, Any, Iterable

import structlog

from skvaider import utils
from skvaider.config import ModelInstanceConfig
from skvaider.manifest import Serial
from skvaider.utils import TaskManager

from .models import AIModel

if TYPE_CHECKING:
    from .backends import Backend

log = structlog.stdlib.get_logger()


class ModelSemaphore:
    """Controls concurrent access to a model across all backends.

    Acts like a semaphore where the total capacity is the sum of
    each backend's per-model limit. Acquiring a slot selects the
    least-busy backend with available capacity.

    """

    model_id: str
    pool: "Pool"
    _condition: asyncio.Condition

    def __init__(self, model_id: str, pool: "Pool"):
        self.model_id = model_id
        self.pool = pool
        self.lock = asyncio.Lock()
        self.released = asyncio.Event()

    def _candidates(
        self, excluded_backends: Iterable[str] = ()
    ) -> list[AIModel]:
        return [
            b.models[self.model_id]
            for b in self.pool.backends
            if self.model_id in b.models
            and b.healthy
            and b.models[self.model_id].is_loaded
            and b.url not in excluded_backends
        ]

    async def acquire(
        self, excluded_backends: Iterable[str] = ()
    ) -> AIModel | None:
        deadline = utils.now() + datetime.timedelta(seconds=120)
        async with self.lock:
            # Ok, we got the lock - we're the next allowed to acquire a free slot.
            # This may take a while and we might not find one right now, but we're
            # still next, so that's why we keep the lock maybe longer.
            while candidates := self._candidates(excluded_backends):
                available = [m for m in candidates if m.in_progress < m.limit]
                if not available:
                    timeout = (deadline - utils.now()).total_seconds()
                    if timeout < 0:
                        return
                    try:
                        log.debug("waiting for limit to sink")
                        await asyncio.wait_for(
                            self.released.wait(), timeout=timeout
                        )
                        self.released.clear()
                        log.debug("request released, checking backends again")
                    except asyncio.TimeoutError:
                        log.debug("timed out waiting for limit")
                        return
                    continue

                    # without this, min() gets the empty list
                    continue  # must re-evaluate available; see test_semaphore_waiter_woken_on_release

                best = min(available, key=lambda m: m.in_progress)
                best.in_progress += 1
                best.idle.clear()
                return best

    async def release(self, model: AIModel) -> None:
        """Release a used slot in a model."""
        model.in_progress -= 1
        if model.in_progress == 0:
            model.idle.set()
        self.released.set()

    @contextlib.asynccontextmanager
    async def use(self, excluded_backends: Iterable[str] = ()):
        model = await self.acquire(excluded_backends)
        try:
            yield model.backend if model else None
        finally:
            if model:
                await self.release(model)


type ModelMap = dict[str, set[str]]


class Pool:
    backends: list["Backend"]
    tasks: TaskManager
    semaphores: dict[str, ModelSemaphore]
    model_configs: dict[str, ModelInstanceConfig]

    # Protect against multiple operations performing larger scale
    # model management tasks like loading/unloading over multiple backends.
    # E.g. if multiple models want to be loaded then they should not start
    # loading/unloading models mixed throughout as this will cause confusion
    model_management_lock: asyncio.Lock

    _last_map: ModelMap
    map_serial: Serial

    def __init__(
        self,
        model_configs: Iterable[ModelInstanceConfig] = (),
        backends: Iterable["Backend"] = (),
    ):
        self.tasks = TaskManager()
        self.semaphores = {}
        self._last_map = {}
        self.map_serial = Serial()

        # Model configurations from config file
        self.model_configs = {m.id: m for m in model_configs}

        self.model_management_lock = asyncio.Lock()

        for model_id in self.model_configs:
            self.semaphores[model_id] = ModelSemaphore(model_id, self)

        self.backends = []
        for backend in backends:
            self.backends.append(backend)
            backend.pool = self
            self.tasks.create(backend.monitor_health_and_update_models)

    def placement_map(self) -> ModelMap:
        """Create a placement map of "which model should go where."

        The idea here is that we have a stable sorting that doesn't flap around too much
        when backends disappear and come back.

        Returns a map of backend IDs -> set of model ids to be loaded there.

        """
        available_resources: dict[
            str, dict[str, int]
        ] = {}  # backend -> resource_type -> total resource
        map: dict[str, set[str]] = {}
        # Fill the usage projection with the total memory.
        for backend in self.backends:
            available_resources[backend.url] = resources = {}
            for resource, params in backend.memory.items():
                resources[resource] = params["total"]
            map[backend.url] = set()

        for model in sorted(
            self.model_configs.values(),
            key=lambda m: m.total_size(),
            reverse=True,
        ):
            unplaced_instances = model.instances
            candidates = [
                b for b in self.backends if b.healthy and model.id in b.models
            ]
            candidates.sort(
                key=lambda b: (
                    b.models[model.id].fit_score(available_resources[b.url]),
                    b.url,
                ),
                reverse=True,
            )
            for backend in candidates:
                use_this_backend = True
                # First pass: check whether this model fits here.
                for resource in available_resources[backend.url]:
                    available = available_resources[backend.url][resource]
                    required = model.memory.get(resource, 0)
                    if required > available:
                        use_this_backend = False
                        break
                if not use_this_backend:
                    continue
                # Make a second pass to update the usage map.
                for resource in available_resources[backend.url]:
                    available_resources[backend.url][resource] -= (
                        model.memory.get(resource, 0)
                    )
                map[backend.url].add(model.id)
                unplaced_instances -= 1
                if not unplaced_instances:
                    break
            if unplaced_instances:
                pass
                # XXX show in monitoring, not continuous logging
                # log.warning(
                #     "Could not place sufficient model instances in pool",
                #     model=model.id,
                #     desired=model.instances,
                #     unplaced=unplaced_instances,
                # )
        return map

    async def rebalance(self) -> None:
        """Rebalance model instances across backends to match desired state."""
        if self.model_management_lock.locked():
            return
        async with self.model_management_lock:
            map = self.placement_map()
            map_changed = map != self._last_map
            if not map_changed:
                return
            self._last_map = map
            self.map_serial.update()
            lines = [f"  serial: {self.map_serial}"] + [
                f"  {url}: {sorted(models)}"
                for url, models in sorted(map.items())
            ]
            log.info("New model distribution map\n" + "\n".join(lines))

            for backend in self.backends:
                if not backend.healthy:
                    continue
                # XXX could also set this as background tasks, but fine for now.
                await backend.update_manifest(map[backend.url], self.map_serial)

    def count_loaded_instances(self, model_id: str) -> int:
        count = 0
        for backend in self.backends:
            if model_id not in backend.models:
                continue
            if backend.models[model_id].is_loaded:
                count += 1
        return count

    def report_map(self) -> dict[Any, Any]:
        """Return a report map of the status of known backends and models"""
        report: dict[Any, Any] = {}
        for backend in self.backends:
            br = report[backend.url] = {}
            br["models"] = {}
            br["memory"] = backend.memory
            for model in backend.models.values():
                br["models"][model.id] = {
                    "loaded": model.is_loaded,
                    "memory": model.memory_usage,
                }
        return report

    def close(self):
        self.tasks.terminate()
