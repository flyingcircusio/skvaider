import asyncio
import datetime
import re
import unicodedata
from collections.abc import Callable, Coroutine
from typing import Any, Generic, Literal, TypeVar

import httpx
import structlog.stdlib
from pydantic import BaseModel, Field

log = structlog.stdlib.get_logger()


def log_task_exception(task: asyncio.Task[Any]) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        pass  # Task cancellation should not be logged as an error.
    except Exception:  # pylint: disable=broad-except
        log.exception("Exception raised by task = %r", task)


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
    """Keep track of tasks.

    Automatically clean up tasks that are done.

    Allow unique tasks where a new task is ignored if a task with the same id already exists.

    Install generic logging callback if exceptions occur.

    Support polling tasks continuously.

    Cancel tasks when cleaning up.

    """

    _tasks: list[asyncio.Task[Any]]
    unique_task_map: dict[str, asyncio.Task[Any]]

    def __init__(self):
        self._tasks = []
        self.unique_task_map = dict()

    @property
    def count(self):
        """Return the number of managed tasks"""
        return len(self._tasks)

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
        id: str = "",
    ) -> asyncio.Task[Any]:
        if id:
            if id in self.unique_task_map:
                return self.unique_task_map[id]

        task = asyncio.create_task(func(*args))
        task.add_done_callback(log_task_exception)
        self._tasks.append(task)

        if id:
            self.unique_task_map[id] = task

            def cleanup_map(t: asyncio.Task[Any]):
                self.unique_task_map.pop(id, None)

            task.add_done_callback(cleanup_map)

        def cleanup_list(t: asyncio.Task[Any]):
            try:
                self._tasks.remove(t)
            except ValueError:
                # Task was already removed (e.g., by terminate())
                pass

        task.add_done_callback(cleanup_list)
        return task

    def cancel(self, id: str):
        if id in self.unique_task_map:
            self.unique_task_map[id].cancel()

    def terminate(self) -> None:
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        self.unique_task_map.clear()


# mockable version
def now():
    return datetime.datetime.now(datetime.UTC)


# tz-aware datetime
datetime_min = datetime.datetime.min.replace(tzinfo=datetime.UTC)


type RequestMethod = (
    Literal["get"]
    | Literal["post"]
    | Literal["patch"]
    | Literal["head"]
    | Literal["delete"]
    | Literal["put"]
)


class ResponseModel(BaseModel):
    pass


T = TypeVar("T")
ResponseModelT = TypeVar("ResponseModelT", bound=ResponseModel)


class RequestModel(BaseModel, Generic[ResponseModelT]):
    request_method: RequestMethod = Field(default="get", exclude=True)
    request_path: str = Field(exclude=True)
    response_model: type[ResponseModelT] = Field(exclude=True)


class ModelAPI:
    base_url: str
    client: httpx.AsyncClient

    def __init__(self, url: str):
        self.base_url = url
        self.client = httpx.AsyncClient(follow_redirects=True)

    async def __call__(
        self,
        request: RequestModel[ResponseModelT],
        method: RequestMethod | None = None,
        timeout: int = 10,
    ) -> ResponseModelT:
        r = await self.client.request(
            method or request.request_method,
            self.base_url + request.request_path,
            json=request.model_dump(
                mode="json",
                exclude={"request_method", "request_path", "response_model"},
            ),
            timeout=timeout,
        )
        r.raise_for_status()
        return request.response_model.model_validate(r.json())
