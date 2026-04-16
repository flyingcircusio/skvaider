import asyncio
from typing import Any, Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field

from skvaider.config import ModelInstanceConfig

from .backends import Backend

log = structlog.stdlib.get_logger()


class CheckResult(BaseModel):
    status: Literal["ok", "warning", "critical"]
    message: str


class AIModel(BaseModel):
    """Model object per backend."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    object: str = "model"
    created: int = 0
    owned_by: str

    log: Any = Field(default=None, exclude=True)

    backend: Backend = Field(exclude=True)
    is_loaded: bool = Field(default=False, exclude=True)
    memory_usage: dict[str, int] = Field(default_factory=dict, exclude=True)
    functional_check: CheckResult | None = Field(default=None, exclude=True)

    limit: int = Field(default=1, exclude=True)
    in_progress: int = Field(default=0, exclude=True)

    # Urks, I hate this.
    idle: asyncio.Event = Field(default_factory=asyncio.Event, exclude=True)

    def __init__(self, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.log = log.bind(model=self.id, backend=self.backend.url)
        self.idle = asyncio.Event()
        self.idle.set()

    @property
    def config(self) -> ModelInstanceConfig:
        return self.backend.pool.model_configs[self.id]

    @property
    def configured_memory(self) -> dict[str, int]:
        """Get the configured memory for this model from pool config."""
        try:
            return self.config.memory
        except KeyError:
            return {}

    def total_size(self) -> int:
        return sum(self.configured_memory.values())

    def check_memory_usage(self) -> dict[str, tuple[int, int]]:
        """Check actual vs configured memory usage.

        Returns a dict of resource -> (actual, configured) for resources
        where actual exceeds configured.
        """
        exceeding: dict[str, tuple[int, int]] = {}
        for resource, actual in self.memory_usage.items():
            configured = self.configured_memory.get(resource, 0)
            if actual > configured:
                exceeding[resource] = (actual, configured)
        return exceeding

    def fit_score(self, resources: dict[str, int] | None = None) -> float:
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

        If `resources` is given, calculate the score based on that, otherwise check
        the backend's free resources.

        """
        score = 0.0
        for resource, usage in self.configured_memory.items():
            if usage == 0:
                continue
            if resources:
                available = resources.get(resource, 0)
            else:
                available = self.backend.memory.get(resource, {}).get("free", 0)
            if self.is_loaded:
                available += usage
            if available < usage:
                continue
            score += 1 - (usage / available)
        return score
