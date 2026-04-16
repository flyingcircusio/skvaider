import asyncio
import functools
import shutil
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, Concatenate, ParamSpec, Protocol, TypeVar

import structlog

from skvaider.inference.model import Model
from skvaider.manifest import Serial
from skvaider.utils import TaskManager

from .resources import (
    MemoryMonitor,
    NvidiaMemoryMonitor,
    RAMMonitor,
)

log = structlog.get_logger()

P = ParamSpec("P")
R = TypeVar("R")


class HasModelLock(Protocol):
    model_lock: asyncio.Lock


SelfT = TypeVar("SelfT", bound=HasModelLock)


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
    """Decorator that acquires self.model_lock before executing an async method."""

    @functools.wraps(func)
    async def wrapper(self: SelfT, *args: P.args, **kwargs: P.kwargs) -> R:
        async with self.model_lock:
            return await func(self, *args, **kwargs)

    return wrapper


class ModelAlreadyLoading(Exception):
    """The model is already loading."""


class Manager:
    models_dir: Path
    models: dict[str, Model]
    monitors: dict[str, MemoryMonitor]
    _tasks: list[asyncio.Task[None]]
    model_lock: asyncio.Lock

    # The manifest manages the intended state for this inference service.
    # At the moment it's the list of model names that should be loaded here.

    manifest_serial: Serial
    _manifest: set[str]
    _manifest_changed: asyncio.Event

    def __init__(self, models_dir: Path, log_dir: Path):
        self.tasks = TaskManager()
        self.models_dir = models_dir
        self.log_dir = log_dir
        self.models = {}
        self.model_lock = asyncio.Lock()
        self._manifest = set()
        self.manifest_serial = Serial.floor()
        self._manifest_changed = asyncio.Event()

        self.monitors = {"ram": RAMMonitor(self)}
        # if shutil.which("rocm-smi"):
        #     self.monitors["rocm"] = ROCmMemoryMonitor(self)
        if shutil.which("nvidia-smi"):
            self.monitors["nvidia"] = NvidiaMemoryMonitor(self)

        for monitor in self.monitors.values():
            self.tasks.poll(monitor.update_global_usage, interval=10)
            self.tasks.poll(monitor.update_model_usage, interval=10)

        self.tasks.create(self.converge)

    def add_model(self, model: Model) -> None:
        assert model.config.id is not None
        assert model.config.id not in self.models
        self.models[model.config.id] = model
        model.datadir = self.models_dir / model.slug
        model.datadir.mkdir(exist_ok=True)
        model.log_dir = self.log_dir

    def list_models(self) -> list[Model]:
        return list(self.models.values())

    @property
    def manifest(self) -> set[str]:
        return self._manifest

    @manifest.setter
    def manifest(self, value: set[str]) -> None:
        self._manifest = set(
            [model_id for model_id in value if model_id in self.models]
        )
        self._manifest_changed.set()

    def update_manifest(self, model_ids: set[str], serial: Serial) -> None:
        if serial <= self.manifest_serial:
            log.info(
                "ignoring manifest with stale serial",
                serial=serial,
                current=self.manifest_serial,
            )
            return
        self.manifest_serial = serial
        self.manifest = model_ids

    async def apply_manifest(self) -> None:
        """Converge once on the manifest."""
        previous_todo = (), ()
        while True:
            running = {
                name
                for name, m in self.models.items()
                if m.process_status in ("running", "starting")
            }
            to_unload = list(running - self.manifest)
            to_load = list(self.manifest - running)
            if not (to_unload or to_load):
                break
            if previous_todo == (to_unload, to_load):
                log.warning(
                    "Not making progress during convergence, giving up."
                )
                break
            previous_todo = (to_unload, to_load)
            if to_unload:
                model_id = to_unload[0]
                try:
                    await self.unload_model(model_id)
                except Exception:
                    log.exception(
                        "Error unloading model during convergence",
                        model=model_id,
                    )
            else:
                model_id = to_load[0]
                try:
                    await self.start_model(model_id)
                except Exception:
                    log.exception(
                        "Error starting model during convergence",
                        model=model_id,
                    )

    async def converge(self) -> None:
        # Running this in a continuous loop helps us avoid starting multiple convergence
        # tasks in parallel if messages come in faster. The apply_manifest() is written
        # in a convergent manner as well and will keep updating towards newer state as
        # needed.
        while True:
            try:
                await self._manifest_changed.wait()
                self._manifest_changed.clear()
                await self.apply_manifest()
            except Exception:
                log.exception(
                    "Unexpected exception in manifest convergence loop"
                )

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
            if "running" not in model.status:
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
