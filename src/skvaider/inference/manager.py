import asyncio
import functools
import shutil
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, Concatenate, ParamSpec, Protocol, TypeVar

import structlog

from skvaider.inference.model import Model
from skvaider.utils import TaskManager

from .resources import (
    MemoryMonitor,
    NvidiaMemoryMonitor,
    RAMMonitor,
)

log = structlog.get_logger()

P = ParamSpec("P")
R = TypeVar("R")


class HasLock(Protocol):
    _lock: asyncio.Lock


SelfT = TypeVar("SelfT", bound=HasLock)


class UserManagerLock:
    def __init__(self):
        self._users = 0
        self._manager_lock = asyncio.Lock()
        self._user_lock = asyncio.Lock()

    async def user_acquire(self):
        async with self._user_lock:
            if self._users == 0:
                # We can only start using this lock if the manager isn't running.
                await self._manager_lock.acquire()
            self._users += 1

    async def user_release(self):
        async with self._user_lock:
            self._users -= 1
            if self._users == 0:
                # The manager now may start again.
                self._manager_lock.release()

    async def manager_acquire(self):
        await self._manager_lock.acquire()

    def manager_release(self):
        self._manager_lock.release()


def locked(
    func: Callable[Concatenate[SelfT, P], Coroutine[Any, Any, R]],
) -> Callable[Concatenate[SelfT, P], Coroutine[Any, Any, R]]:
    """Decorator that acquires self._lock before executing an async method."""

    @functools.wraps(func)
    async def wrapper(self: SelfT, *args: P.args, **kwargs: P.kwargs) -> R:
        async with self._lock:  # pyright: ignore[reportPrivateUsage]
            return await func(self, *args, **kwargs)

    return wrapper


class ModelAlreadyLoading(Exception):
    """The model is already loading."""


class Manager:
    models_dir: Path
    models: dict[str, Model]
    monitors: dict[str, MemoryMonitor]
    _tasks: list[asyncio.Task[None]]

    def __init__(self, models_dir: Path, log_dir: Path | None = None):
        self.tasks = TaskManager()
        self.models_dir = models_dir
        self.log_dir = log_dir
        self.models = {}
        self._lock = asyncio.Lock()

        self.monitors = {"ram": RAMMonitor(self)}
        # if shutil.which("rocm-smi"):
        #     self.monitors["rocm"] = ROCmMemoryMonitor(self)
        if shutil.which("nvidia-smi"):
            self.monitors["nvidia"] = NvidiaMemoryMonitor(self)

        for monitor in self.monitors.values():
            self.tasks.poll(monitor.update_global_usage, interval=10)
            self.tasks.poll(monitor.update_model_usage, interval=10)

    def add_model(self, model: Model) -> None:
        assert model.config.id is not None
        assert model.config.id not in self.models
        self.models[model.config.id] = model
        model.datadir = self.models_dir / model.slug
        model.datadir.mkdir(exist_ok=True)
        model.log_dir = self.log_dir

    def list_models(self) -> list[Model]:
        return list(self.models.values())

    @locked
    async def start_model(
        self,
        model_name: str,
        timeout: int = 600,  # XXX the timeout might need to be model specific? and might need to be communicated to the gateway?
    ) -> Model:
        model = self.models[model_name]
        if model.lock.manager_locked():
            # I'm relatively sure this is happening atomically:
            # https://stackoverflow.com/questions/74923841/asyncio-try-to-acquire-a-lock-without-waiting-on-it
            raise ModelAlreadyLoading()
        await model.lock.manager_acquire()
        try:
            if "active" not in model.status:
                try:
                    await asyncio.wait_for(model.start(), timeout=timeout)
                except asyncio.TimeoutError:
                    log.error("Timeout starting model", model=model_name)
                    await model.terminate()
                    raise
                # XXX let this trigger on the monitor via the event handler?
                for monitor in self.monitors.values():
                    await monitor.update_global_usage()
        finally:
            model.lock.manager_release()
        return model

    async def use_model(
        self,
        model_name: str,
    ) -> Model | None:
        model = self.models.get(model_name)
        if not model:
            return
        if "active" not in model.status:
            return
        return model

    @locked
    async def unload_model(self, model_name: str) -> None:
        model = self.models[model_name]
        await model.lock.manager_acquire()
        try:
            if model.process_status in ["running", "starting"]:  # idempotent
                await model.terminate()
                # XXX let this trigger on the monitor via the event handler?
                for monitor in self.monitors.values():
                    await monitor.update_global_usage()
        finally:
            model.lock.manager_release()

    async def shutdown(self) -> None:
        self.tasks.terminate()
        for model in list(self.models.values()):
            await model.lock.manager_acquire()
            try:
                await model.terminate()
            finally:
                model.lock.manager_release()

        # It would be cleaner if we'd use a lock here, but shutdown otherwise
        # can end up locked infinitely it seems.
        self.models.clear()
