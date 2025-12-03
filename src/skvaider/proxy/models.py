import asyncio
import contextlib
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import structlog
from pydantic import BaseModel, ConfigDict, Field

from .backends import Backend

log = structlog.stdlib.get_logger()

T = TypeVar("T")


class AIModel(BaseModel):
    """Model object per backend."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    object: str = "model"
    created: int = 0
    owned_by: str

    backend: Backend = Field(exclude=True)
    last_used: float = Field(default=0, exclude=True)
    in_progress: int = Field(default=0, exclude=True)
    limit: int = Field(default=5, exclude=True)
    idle: asyncio.Event = Field(default=True, exclude=True)
    is_loaded: bool = Field(default=False, exclude=True)
    memory_usage: int = Field(default=0, exclude=True)
    log: Any = Field(default=None, exclude=True)

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.idle = asyncio.Event()
        self.idle.set()

        self.log = log.bind(model=self.id, backend=self.backend.url)

    @contextlib.asynccontextmanager
    async def use(self):
        try:
            yield
        finally:
            self.in_progress -= 1
            self.log.debug("done", in_progress=self.in_progress)
            if not self.in_progress:
                self.log.debug("idling")
                self.idle.set()

    async def wait(self):
        await self.idle.wait()
        return self


class ListResponse(BaseModel, Generic[T]):
    object: str = "list"
    data: list[T]
