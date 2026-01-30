import asyncio
import re
import unicodedata
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

import structlog.stdlib

T = TypeVar("T")

log = structlog.stdlib.get_logger()


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


def slugify(text: str, max_length: int = 255) -> str:
    """
    Convert text to a slug safe for Linux, Windows, and URL path components.

    - Normalizes unicode and converts to ASCII
    - Lowercases everything
    - Replaces spaces/separators with hyphens
    - Removes unsafe characters
    - Avoids Windows reserved names

    """
    # Normalize unicode to ASCII equivalents (é -> e, etc.)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    # Lowercase
    text = text.lower()

    # Replace common separators with hyphens
    text = re.sub(r"[\s:_./\\]+", "-", text)

    # Remove anything that isn't alphanumeric or hyphen
    text = re.sub(r"[^a-z0-9-]", "", text)

    # Collapse multiple hyphens
    text = re.sub(r"-+", "-", text)

    text = text.strip("-")

    input_len = len(text)
    if (overflow := input_len - max_length) > 0:
        # This is tricky: the rounding works out exactly so
        # that on uneven numbers we'll remove (one more) character on the middle
        # or left and on even numbers one more to the right
        # fmt: off
        text = (text[:int(input_len/2-overflow/2)] +
                text[int(input_len/2+overflow/2):])
        # fmt: on

    # Collapse multiple hyphens again
    text = re.sub(r"-+", "-", text)

    # Handle empty result
    if not text:
        text = "unnamed"

    return text


class TaskManager:
    _tasks: list[asyncio.Task[None]]

    def __init__(self):
        self._tasks = []

    def poll(
        self,
        func: Callable[..., Coroutine[Any, Any, None]],
        *args: Any,
        interval: float,
    ) -> None:
        """Add a function to be polled. Uses @poll decorator metadata."""

        async def loop() -> None:
            while True:
                try:
                    await func(*args)
                except Exception:
                    log.exception(f"An error occured polling {func!r}")
                await asyncio.sleep(interval)

        self.create(loop)

    def create(
        self,
        func: Callable[..., Coroutine[Any, Any, Any]],
        args: Any = (),
    ) -> asyncio.Task[Any]:
        self._tasks.append(t := asyncio.create_task(func(*args)))
        return t

    def terminate(self) -> None:
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
