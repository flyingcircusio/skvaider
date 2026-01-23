# this is duplicated to avoid invalid cross-references between the aramaki and
# skvaider namespace and easier library extraction

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

import structlog.stdlib

log = structlog.stdlib.get_logger()

T = TypeVar("T")


def log_task_exception(task: asyncio.Task[Any]) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        pass  # Task cancellation should not be logged as an error.
    except Exception:  # pylint: disable=broad-except
        log.exception("Exception raised by task = %r", task)


def create_task(aw: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
    t = asyncio.create_task(aw)
    t.add_done_callback(log_task_exception)
    return t
