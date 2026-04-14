import time
from typing import Any

from pydantic import BaseModel, GetCoreSchemaHandler
from pydantic_core import core_schema


class Serial:
    """Placement manifest versioning: a microsecond-epoch hex stamp plus a monotonic counter.

    The stamp is a 10-char zero-padded hex encoding of the time when the serial is initialised
    (usually during server startup) in microseconds (40-bit).

    Using a fixed-width hex string means lexicographic order equals chronological order, so
    serials from any two proxy instances are always comparable: a restarted proxy will always
    produce a later stamp than its predecessor.

    The integer counter lets a single proxy instance mark successive map
    revisions without needing sub-microsecond precision and making it easier to distinguish
    server restarts versus a large number of manifest changes.

    """

    generation: str
    serial: int

    def __init__(self) -> None:
        now = time.time_ns()
        now_40 = now >> now.bit_length() - 40
        now_hex = f"{now_40:010x}"
        self.generation = now_hex[:5] + "-" + now_hex[5:]
        self.serial = 0

    @classmethod
    def floor(cls) -> "Serial":
        """Return the lowest possible serial, used as an initial sentinel value."""
        obj = cls.__new__(cls)
        obj.generation = ""
        obj.serial = -1
        return obj

    @classmethod
    def from_json(cls, data: tuple[str, int]) -> "Serial":
        """Construct from the ``(generation, serial)`` pair carried over the wire."""
        obj = cls.__new__(cls)
        obj.generation = data[0]
        obj.serial = data[1]
        return obj

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        def validate(v: object) -> "Serial":
            if isinstance(v, cls):
                return v
            if isinstance(v, (list, tuple)) and len(v) == 2:  # pyright: ignore[reportUnknownArgumentType]
                return cls.from_json((str(v[0]), int(v[1])))  # pyright: ignore[reportUnknownArgumentType]
            raise ValueError(f"Cannot coerce {v!r} to Serial")

        return core_schema.no_info_plain_validator_function(
            validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda v: list(v.as_tuple()),
                info_arg=False,
            ),
        )

    def update(self) -> None:
        """Increment the serial to mark a new map version."""
        self.serial += 1

    def as_tuple(self) -> tuple[str, int]:
        return (self.generation, self.serial)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Serial):
            return NotImplemented
        return (
            self.generation == other.generation and self.serial == other.serial
        )

    def __hash__(self) -> int:
        return hash(self.as_tuple())

    def __lt__(self, other: "Serial") -> bool:
        return (self.generation, self.serial) < (other.generation, other.serial)

    def __le__(self, other: "Serial") -> bool:
        return (self.generation, self.serial) <= (
            other.generation,
            other.serial,
        )

    def __str__(self) -> str:
        return f"{self.generation}/{self.serial}"

    def __repr__(self) -> str:
        return f"Serial({self})"


class ManifestRequest(BaseModel):
    """Wire format for the manifest PATCH request sent from proxy to inference server."""

    models: set[str]
    serial: Serial
