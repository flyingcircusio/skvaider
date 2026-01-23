import asyncio
import contextlib
from typing import TYPE_CHECKING

import structlog

from skvaider import utils

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
    queues: dict[str, asyncio.Queue[ProxyRequest]]  # one queue per model

    def __init__(self):
        self.backends = []
        self.health_check_tasks = []
        self.queues = {}
        self.models: dict[str, AIModel] = {}
        self.queue_tasks: dict[str, asyncio.Task[None]] = {}

    def add_backend(self, backend: "Backend"):
        self.backends.append(backend)
        self.health_check_tasks.append(
            utils.create_task(backend.monitor_health_and_update_models(self))
        )

    def update_model_maps(self):
        # XXX the same model must not be owned by different organizations!
        # This requires a bit more thought how to handle consistency if
        # backends answer with conflicting/differing model data.
        # XXX The models in self.models in pool still have backreferences to individual backends
        self.models.clear()
        for backend in self.backends:
            self.models.update(backend.models)

        # Add new models
        for model_id in self.models:
            if model_id in self.queues:
                continue
            # Here, a new model appeared
            self.queues[model_id] = asyncio.Queue()
            self.queue_tasks[model_id] = utils.create_task(
                self.assign_backends(model_id)
            )

        # ensure model is loaded on at least one backend
        for model_id in self.models:
            utils.create_task(self.warm_up_model(model_id))

        # Remove outdated model queues and tasks
        for model_id, task in self.queue_tasks.items():
            if model_id in self.models:
                continue
            task.cancel()
            del self.queue_tasks[model_id]

        for model_id in self.queues:
            if model_id in self.models:
                continue
            del self.queues[model_id]

    async def warm_up_model(self, model_id: str):
        if any(
            model_id in b.models and b.models[model_id].is_loaded
            for b in self.backends
        ):
            return
        # load model on backend with least memory usage
        candidate_backends = [b for b in self.backends if model_id in b.models]
        if not candidate_backends:
            log.error(
                "trying to load model not present on any backend {model}, are the backends flapping?",
                model=model_id,
            )
            return
        candidate_backends.sort(key=lambda b: b.memory_usage)
        backend = candidate_backends[0]
        log.info(
            "loading model on backend", model=model_id, backend=backend.url
        )
        await backend.load_model_with_options(model_id)

    async def assign_backends(self, model_id: str):
        """Continuously assign requests to backends.

        Perform batching and model distribution and warmup.

        """
        while True:
            log.debug("waiting for request", model=model_id)
            queue = self.queues[model_id]
            request_batch = [await queue.get()]
            log.debug("got request", model=model_id)

            # Now, are there any backends with the model loaded and are they available?
            while not (
                model_backends := [
                    b for b in self.backends if model_id in b.models
                ]
            ):
                log.warning("no backends with model available", model=model_id)
                await asyncio.sleep(1)

            loaded_backends = [
                b for b in model_backends if b.models[model_id].is_loaded
            ]
            idle_backends = [
                b for b in loaded_backends if b.models[model_id].idle
            ]
            not_loaded_backends = [
                b for b in model_backends if not b.models[model_id].is_loaded
            ]

            if (
                not idle_backends
                and len(loaded_backends) < 2
                and not_loaded_backends
            ):  # At most 2 instances per model
                # Load the model on a host with as little used memory as possible
                # if we have spare hosts.
                not_loaded_backends.sort(key=lambda b: b.memory_usage)
                new_backend = not_loaded_backends[0]
                log.debug(
                    "warming up model on new backend",
                    backend=new_backend.url,
                    model=model_id,
                )
                await new_backend.load_model_with_options(model_id)
                idle_backends.insert(0, new_backend)

            if not idle_backends:
                # Need to wait for an idle backend
                log.debug("waiting for idle backends", model=model_id)
                tasks = [
                    utils.create_task(b.models[model_id].wait())
                    for b in model_backends
                ]
                ready_tasks, _ = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                idle_backends = [t.result().backend for t in ready_tasks]
            backend = idle_backends[0]
            model = backend.models[model_id]
            log.debug("got idle backend", backend=backend.url, model=model_id)

            # This should not be necessary, but it should also be gratuitous.

            await backend.load_model_with_options(model_id)
            log.debug("gathering more batchable requests", model=model_id)
            # Prime the model
            # Wait up to 0.1s for up to N requests
            more_request_tasks = await asyncio.gather(
                *[
                    asyncio.wait_for(queue.get(), 0.001)
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
            for request in request_batch:
                log.debug(
                    "assigning request to backend",
                    model=model_id,
                    backend=backend.url,
                )
                model.in_progress += 1
                request.model = model
                request.backend_available.set()
            model.idle = False

    def close(self):
        for task in self.health_check_tasks:
            task.cancel()
        for task in self.queue_tasks.values():
            task.cancel()

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
