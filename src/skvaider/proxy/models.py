import asyncio
import contextlib
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field

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
    last_used: float = Field(default=0, exclude=True)
    in_progress: int = Field(default=0, exclude=True)
    limit: int = Field(default=5, exclude=True)
    is_loaded: bool = Field(default=False, exclude=True)
    memory_usage: int = Field(default=0, exclude=True)
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
        if not value:
            self.__idle.clear()

    @contextlib.asynccontextmanager
    async def use(self):
        try:
            yield
        finally:
            self.in_progress -= 1
            self.log.debug("done", in_progress=self.in_progress)
            if not self.in_progress:
                self.log.debug("idling")
                self.__idle.set()

    async def wait(self):
        await self.__idle.wait()
        return self
