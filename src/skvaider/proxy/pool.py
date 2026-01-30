import asyncio
import contextlib
from typing import TYPE_CHECKING

import structlog

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

    def __init__(self):
        self.backends = []
        self.tasks = TaskManager()
        self.queues = {}
        self.models: dict[str, AIModel] = {}

    def add_backend(self, backend: "Backend"):
        self.backends.append(backend)
        self.tasks.create(
            backend.monitor_health_and_update_models, args=(self,)
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
            self.tasks.create(
                self.assign_backends, args=(model_id,), id=f"{model_id}:queue"
            )

        # ensure model is loaded on at least one backend
        for model_id in self.models:
            self.warm_up_model(model_id)

        for task_id in self.tasks.unique_task_map.keys():
            model_id, _ = task_id.split(":")
            if model_id in self.models:
                continue
            self.tasks.cancel(task_id)

    def warm_up_model(self, model_id: str):
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

    async def add_model_instance(self, model_id: str):
        # add another model instance on a backend with least memory usage and where its not running, yet.
        candidate_backends = [
            (b, b.models[model_id].fit_score())
            for b in self.backends
            if model_id in b.models and not b.models[model_id].is_loaded
        ]
        # Eliminate backends where the score is 0
        candidate_backends = list(
            filter(lambda b: b[1] > 0, candidate_backends)
        )
        if not candidate_backends:
            log.error("no backend available to load model", model=model_id)
            return
        candidate_backends.sort(key=lambda b: b[1], reverse=True)
        for b, score in candidate_backends:
            log.debug("model eval", model=model_id, backend=b.url, score=score)

        backend, _ = candidate_backends[0]
        await backend.load_model_with_options(model_id, self)

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
