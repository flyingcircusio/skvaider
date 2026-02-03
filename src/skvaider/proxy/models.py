import asyncio
import contextlib
import datetime
from typing import Any, Self

import structlog
from pydantic import BaseModel, ConfigDict, Field

from skvaider import utils

from .backends import Backend

log = structlog.stdlib.get_logger()


class AIModel(BaseModel):
    """Model object per backend."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    object: str = "model"
    created: int = 0
    owned_by: str

    backend: Backend = Field(exclude=True)
    last_used: datetime.datetime = Field(
        default=utils.datetime_min, exclude=True
    )
    in_progress: int = Field(default=0, exclude=True)
    limit: int = Field(default=5, exclude=True)
    is_loaded: bool = Field(default=False, exclude=True)
    memory_usage: dict[str, int] = Field(default_factory=dict, exclude=True)
    log: Any = Field(default=None, exclude=True)

    __idle: asyncio.Event

    def __init__(self, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.__idle = asyncio.Event()
        self.__idle.set()

        self.log = log.bind(model=self.id, backend=self.backend.url)

    @property
    def idle(self) -> bool:
        return self.__idle.is_set()

    @idle.setter
    def idle(self, value: bool) -> None:
        if value:
            self.__idle.set()
        else:
            self.__idle.clear()

    async def wait_for_idle(self) -> Self:
        await self.__idle.wait()
        return self

    def total_size(self) -> int:
        return sum(self.memory_usage.values())

    def fit_score(self) -> float:
        """A score (higher) is better how well this model fits or would fit on it's backend
        compared to the same model on other backends.

        The score is intended to be the same independent of whether it's already been
        loaded or not. The score computes the inverted % of the free memory this would consume,
        so higher values are better.

        This means we prefer to put models on backends that have the most free space left
        after placing the model there.

        It's a sum of normalized values how well it fits in each category of memory. If it doesn't
        use a specific kind of memory then this is not included in the fitting - as the fittings
        can only ever be compared between the same models not against other models.

        If no data is available or the model doesn't fit, we return 0 as to not load this
        model until we have sufficient data.

        """
        score = 0.0
        for backend, usage in self.memory_usage.items():
            if usage == 0:
                continue
            available = self.backend.memory.get(backend, {}).get("free", 0)
            if self.is_loaded:
                available += usage
            if available < usage:
                continue
            score += 1 - (usage / available)
        return score

    def fits_generally(self) -> bool:
        """Check whether this model fits on this backend at all."""
        for backend, usage in self.memory_usage.items():
            if usage == 0:
                continue
            total = self.backend.memory.get(backend, {}).get("total", 0)
            if total <= usage:
                return False
        return True

    @contextlib.asynccontextmanager
    async def use(self):
        try:
            self.last_used = utils.now()
            yield
        finally:
            self.last_used = utils.now()
            self.in_progress -= 1
            self.log.debug("done", in_progress=self.in_progress)
            if not self.in_progress:
                self.log.debug("idling")
                self.idle = True

    async def wait(self):
        await self.__idle.wait()
        return self
