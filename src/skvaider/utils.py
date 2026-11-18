import asyncio
import datetime
import re
import unicodedata
from collections.abc import Callable, Coroutine, Iterator
from typing import Any, Literal, TypeVar, TypeVarTuple, cast

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


TaskArgs = TypeVarTuple("TaskArgs")


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
        self.unique_task_map = {}

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
        func: Callable[[*TaskArgs], Coroutine[Any, Any, Any]],
        args: tuple[*TaskArgs] = (),
        id: str = "",
    ) -> asyncio.Task[Any]:
        if id and id in self.unique_task_map:
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


type RequestMethod = Literal["get", "post", "patch", "head", "delete", "put"]


class ResponseModel(BaseModel):
    pass


T = TypeVar("T")
ResponseModelT = TypeVar("ResponseModelT", bound=ResponseModel)


class RequestModel[ResponseModelT: ResponseModel](BaseModel):
    request_method: RequestMethod = Field(default="get", exclude=True)
    request_path: str = Field(exclude=True)
    response_model: type[ResponseModelT] = Field(exclude=True)


class HealthCheckResult(BaseModel):
    """Outcome of a single health check, including its recent history.

    ``ok`` is the current verdict. ``checked_at`` is when the check last
    ran. ``failure_message`` describes the most recent failure,
    ``failed_at`` when that failure started and ``recovered_at`` when the
    check recovered from it -- all three are kept after a recovery so the
    history of a flapping check stays visible. ``consecutive_failures``
    counts the current streak (0 while the check passes).

    :meth:`merge` carries that history over from the previous result.
    """

    ok: bool = True
    failure_message: str = ""
    checked_at: datetime.datetime | None = None
    failed_at: datetime.datetime | None = None
    recovered_at: datetime.datetime | None = None
    consecutive_failures: int = 0

    @classmethod
    def passed(cls) -> "HealthCheckResult":
        return cls()

    @classmethod
    def failed(cls, message: str) -> "HealthCheckResult":
        return cls(ok=False, failure_message=message)

    @classmethod
    def merge(
        cls,
        previous: "HealthCheckResult | None",
        fresh: "HealthCheckResult",
        now: datetime.datetime | None = None,
    ) -> "HealthCheckResult":
        """Record a fresh result, carrying the history over from ``previous``."""
        now = now or datetime.datetime.now(tz=datetime.UTC)
        result = fresh.model_copy(update={"checked_at": now})
        if previous is None:
            if not result.ok:
                result.failed_at = now
                result.consecutive_failures = 1
            return result

        was_failing = not previous.ok
        result.failed_at = previous.failed_at
        result.recovered_at = previous.recovered_at
        if result.ok:
            result.consecutive_failures = 0
            result.failure_message = previous.failure_message
            if was_failing:
                result.recovered_at = now
        elif was_failing:
            result.failed_at = previous.failed_at or now
            result.consecutive_failures = previous.consecutive_failures + 1
        else:
            result.failed_at = now
            result.consecutive_failures = 1
        return result


class ModelHealth(BaseModel):
    """Health state of an inference model.

    Every check is a typed attribute; ``None`` means the check did not run
    yet or does not apply to the model's task. The overall ``status`` is
    derived from the checks that are present. Checks record their result
    through :meth:`HealthCheckResult.merge`, which keeps their history.
    """

    status: Literal["healthy", "unhealthy", ""] = ""
    status_changed_at: datetime.datetime | None = None

    quick_health_check: HealthCheckResult | None = None
    deep_check_failure: HealthCheckResult | None = None
    completion: HealthCheckResult | None = None
    completion_tool_call: HealthCheckResult | None = None
    completion_with_image_input: HealthCheckResult | None = None
    embedding: HealthCheckResult | None = None
    numerical: HealthCheckResult | None = None

    @classmethod
    def check_names(cls) -> tuple[str, ...]:
        """Names of the check attributes, derived from the schema."""
        return tuple(
            name
            for name in cls.model_fields
            if name not in ("status", "status_changed_at")
        )

    def present_checks(self) -> Iterator[tuple[str, HealthCheckResult]]:
        """Yield the name and result of every check that has run."""
        for name in self.check_names():
            check = cast(HealthCheckResult | None, getattr(self, name))
            if check is not None:
                yield name, check

    def has_failures(self) -> bool:
        return any(not check.ok for _, check in self.present_checks())

    def update_status(self, now: datetime.datetime | None = None) -> bool:
        """Re-derive the status from the present checks.

        Returns True when the status changed (``status_changed_at`` is
        updated in that case).
        """
        now = now or datetime.datetime.now(tz=datetime.UTC)
        present = False
        failures = False
        for _, check in self.present_checks():
            present = True
            if not check.ok:
                failures = True
        if not present:
            new_status: Literal["healthy", "unhealthy", ""] = ""
        elif failures:
            new_status = "unhealthy"
        else:
            new_status = "healthy"
        changed = new_status != self.status
        self.status = new_status
        if changed:
            self.status_changed_at = now
        return changed

    def reset(self, now: datetime.datetime | None = None) -> None:
        """Drop all checks and the status, e.g. when the process stops."""
        for name in self.check_names():
            setattr(self, name, None)
        self.status = ""
        self.status_changed_at = now or datetime.datetime.now(tz=datetime.UTC)


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


class DownloadProgress:
    response: httpx.Response
    size: int = 0
    status: int = 0

    def __init__(self, response: httpx.Response):
        self.response = response
        self.size = int(self.response.headers.get("content-length", 0))

    @property
    def progress(self):
        return int(self.status / self.size) * 100
