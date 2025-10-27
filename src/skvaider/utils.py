import asyncio

import structlog.stdlib

log = structlog.stdlib.get_logger()


def log_task_exception(task: asyncio.Task) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        pass  # Task cancellation should not be logged as an error.
    except Exception:  # pylint: disable=broad-except
        log.exception("Exception raised by task = %r", task)


def create_task(aw):
    t = asyncio.create_task(aw)
    t.add_done_callback(log_task_exception)
    return t
