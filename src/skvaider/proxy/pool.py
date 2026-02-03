import asyncio
import contextlib
import datetime
from typing import TYPE_CHECKING, Any

import structlog

from skvaider import utils
from skvaider.utils import TaskManager

from .models import AIModel

if TYPE_CHECKING:
    from .backends import Backend

log = structlog.stdlib.get_logger()


class ProxyRequest:
    backend_available: asyncio.Event
    model: AIModel | None = None

    def __init__(self):
        self.backend_available = asyncio.Event()


class Pool:
    backends: list["Backend"]
    health_check_tasks: list[asyncio.Task[None]]
    tasks: TaskManager
    queues: dict[str, asyncio.Queue[ProxyRequest]]  # one queue per model

    # Protect against multiple operations performing larger scale
    # model management tasks like loading/unloading over multiple backends.
    # E.g. if multiple models want to be loaded then they should not start
    # loading/unloading models mixed throughout as this will cause confusion
    model_management_lock: asyncio.Lock

    def __init__(self):
        self.backends = []
        self.tasks = TaskManager()
        self.queues = {}

        # This keeps track of the globally known model IDs
        self.models: set[str] = set()

        self.model_management_lock = asyncio.Lock()

    def add_backend(self, backend: "Backend"):
        self.backends.append(backend)
        self.tasks.create(backend.monitor_health_and_update_models)

    def update_model_maps(self):
        # XXX the same model must not be owned by different organizations!
        # This requires a bit more thought how to handle consistency if
        # backends answer with conflicting/differing model data.

        current_backend_models: set[str] = set()
        for backend in self.backends:
            current_backend_models.update(backend.models.keys())

        new_models = current_backend_models - self.models
        deleted_models = self.models - current_backend_models

        # Add new models
        for model_id in new_models:
            self.queues[model_id] = asyncio.Queue()
            self.tasks.create(
                self.assign_backends, args=(model_id,), id=f"{model_id}:queue"
            )

        # Cleanup after deleted models
        for task_id in self.tasks.unique_task_map.keys():
            model_id, _ = task_id.split(":")
            if model_id in deleted_models:
                self.tasks.cancel(task_id)
        for model_id in deleted_models:
            del self.queues[model_id]

        self.models = current_backend_models

        # Ensure reserved instances for remaining models
        for model_id in self.models:
            self.ensure_reserved_instance(model_id)

    def ensure_reserved_instance(self, model_id: str):
        """Ensure there is at least 1 instance of this model running."""
        if any(
            model_id in b.models and b.models[model_id].is_loaded
            for b in self.backends
        ):
            return
        self.tasks.create(
            self.add_model_instance,
            args=(model_id,),
            id=f"{model_id}:add_model_instance",  # needs to be consistent with continuous assignment
        )

        # XXX testing code
        # async def trigger_unload():
        #     await asyncio.sleep(20)
        #     log.warning("unloading models")
        #     for backend in self.backends:
        #         for model_id in backend.models:
        #             log.warning(f"unloading {backend.url} {model_id}")
        #             await backend.unload_model(model_id)

        # self.tasks.create(trigger_unload)

    def find_backends_for_new_model_instance(
        self, model_id: str
    ) -> list["Backend"]:
        """Find backends that could host a new instance of a given model.

        Backends that already run the model are excluded and backends that do not
        have sufficient space are excluded, too.

        Return a list of backends ordered by best fitness first.

        """
        candidate_backends = [
            (b, b.models[model_id].fit_score())
            for b in self.backends
            if model_id in b.models
            and not b.models[model_id].is_loaded
            and b.models[model_id].fits_generally()
        ]
        # Eliminate backends where the score is 0
        candidate_backends = list(
            filter(lambda b: b[1] > 0, candidate_backends)
        )
        candidate_backends.sort(key=lambda b: b[1], reverse=True)
        for b, score in candidate_backends:
            log.debug("model eval", model=model_id, backend=b.url, score=score)
        return [b[0] for b in candidate_backends]

    async def add_model_instance(self, model_id: str):
        """Try to add a(nother) instance of this model.

        WARNING: This method should be called using the task manager with a
        unique id as it doesn't make sense to have it called multiple times
        concurrently.

        Models are only placed on servers where the model is not yet loaded.

        We prefer putting models on servers that have the best fitness (see `AIModel.fit_score()`).

        If there is no room, then we will try to make room.

        We retry up to 3 times, but give up after that.

        """
        retry = 0
        async with self.model_management_lock:
            candidate_backends = self.find_backends_for_new_model_instance(
                model_id
            )
            while not candidate_backends and retry < 3:
                retry += 1
                await self.make_room(model_id)
                candidate_backends = self.find_backends_for_new_model_instance(
                    model_id
                )

            if not candidate_backends:
                log.error("no backend available to load model", model=model_id)
                return
            await candidate_backends[0].load_model(model_id)

    async def make_room(self, model_id: str):
        """Make room in the cluster to fit the given model.

        This likely always should be called while holding the model management lock.

        """
        candidate_backends = [
            b
            for b in self.backends
            if model_id in b.models
            and not b.models[model_id].is_loaded
            and b.models[model_id].fits_generally()
        ]
        candidate_backends.sort(
            key=lambda b: b.models[model_id].fit_score(), reverse=True
        )
        # We could assume the fitness is 0, but maybe something changed in between,
        # so we can just check that.
        for backend in candidate_backends:
            while True:
                if backend.models[model_id].fit_score() > 0:
                    # There should be sufficient space now!
                    return

                def loaded_instances(model_id: str) -> int:
                    counter = 0
                    for backend in self.backends:
                        if model_id not in backend.models:
                            continue
                        if backend.models[model_id].is_loaded:
                            counter += 1
                    return counter

                # Try to find a model that has
                # - more than 1 instance
                # - has not been used in the last 60 seconds
                now = utils.now()
                unload_candidates = [
                    m
                    for m in backend.models.values()
                    if m.is_loaded
                    and loaded_instances(m.id) > 1
                    and (now - m.last_used > datetime.timedelta(seconds=60))
                ]

                if not unload_candidates:
                    # No progress to be made here.
                    break

                # try to unload smaller models first. This is likely not a perfect strategy
                # but avoids unnecessarily unloading very large models.

                unload_candidates.sort(key=lambda m: m.total_size())
                await backend.unload_model(unload_candidates[0].id)

    async def assign_backends(self, model_id: str):
        """Continuously assign requests to backends.

        All requests for this model are seen here in a serialized
        fashion and then we send them off for processing outside
        of this loop.

        At the same time we peek forward into the buffer to ensure
        we can send off multiple requests to the same backend quickly
        after each other to help with batching on the backends, too.

        """
        while True:
            log.debug("waiting for request", model=model_id)
            queue = self.queues[model_id]
            request_batch = [await queue.get()]

            log.debug("got request", model=model_id)

            # 1. Try up to 3 seconds to locate a free backend. As we're in a serialized loop here
            # we don't have to pay attention whether other requests are coming in for now.
            candidates = [
                b.models[model_id]
                for b in self.backends
                if model_id in b.models
            ]

            log.info("waiting for first idle backend", model=model_id)
            done, pending = await asyncio.wait(
                [asyncio.create_task(b.wait_for_idle()) for b in candidates],
                timeout=3,
                return_when=asyncio.FIRST_COMPLETED,
            )

            for p in pending:
                p.cancel()

            if not done:
                log.info(
                    "no idle backend found, starting another instance",
                    model=model_id,
                )
                # 2. We didn't get an idle backend within 3 seconds so we start another one
                self.tasks.create(
                    self.add_model_instance,
                    args=(model_id,),
                    id=f"{model_id}:add_model_instance",  # needs to be consistent with warmup
                )
                log.info("waiting for idle backend (2)", model=model_id)
                # 3. And now we wait longer. There's a timeout to ensure we do not
                # get stuck infinitely here, but loading models can take time.
                done, pending = await asyncio.wait(
                    [
                        asyncio.create_task(b.wait_for_idle())
                        for b in candidates
                    ],
                    timeout=120,
                    return_when=asyncio.FIRST_COMPLETED,
                )

            try:
                model = list(done)[0].result()
            except Exception:
                log.exception(
                    "An error occured waiting for an idle backend, starting over."
                )
                for r in request_batch:
                    await self.queues[model_id].put(r)
                continue
            log.debug(
                "got idle backend", backend=model.backend.url, model=model_id
            )

            log.debug("gathering more batchable requests", model=model_id)
            # Wait a bit to gather more requests that might have piled up.
            more_request_tasks = await asyncio.gather(
                *[
                    asyncio.wait_for(queue.get(), 0.05)
                    for _ in range(model.limit - 1)
                ],
                return_exceptions=True,
            )
            request_batch.extend(
                [
                    t
                    for t in more_request_tasks
                    if not isinstance(t, BaseException)
                ]
            )
            log.debug("got request batch", size=len(request_batch))
            for request in request_batch:
                log.debug(
                    "assigning request to backend",
                    model=model_id,
                    backend=model.backend.url,
                )
                model.in_progress += 1
                model.idle = False
                request.model = model
                request.backend_available.set()

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
                    "last-used": model.last_used,
                    "memory": model.memory_usage,
                }
        return report

    def close(self):
        self.tasks.terminate()

    @contextlib.asynccontextmanager
    async def use(self, model_id: str):
        request = ProxyRequest()
        assert model_id in self.queues
        log.debug("queueing request", model=model_id)
        queue = self.queues[model_id]
        await queue.put(request)
        log.debug("waiting for backend to become available", model=model_id)
        await request.backend_available.wait()
        assert request.model is not None
        log.debug(
            "got backend", backend=request.model.backend.url, model=model_id
        )
        async with request.model.use():
            yield request.model.backend
