import re
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, field_validator


def parse_size(value: str | int) -> int:
    """Parse a human-readable size string into bytes.

    Supports SI units: K, M, G, T (and KB, MB, GB, TB variants).
    Examples: "8G", "8GB", "500M", "1T", "1024"
    """
    if isinstance(value, int):
        return value

    value = value.strip().upper()
    match = re.match(r"^(\d+(?:\.\d+)?)\s*([KMGT])?B?$", value)
    if not match:
        raise ValueError(f"Invalid size format: {value}")

    number = float(match.group(1))
    unit = match.group(2)

    multipliers = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    multiplier = multipliers.get(unit, 1)

    return int(number * multiplier)


class ModelInstanceConfig(BaseModel):
    id: str
    instances: int
    memory: dict[str, int]  # e.g. {"ram": "8G"} or {"rocm-vram": "8G"}
    limit: int = 0
    task: Literal["chat", "embedding"]

    @field_validator("memory", mode="before")
    @classmethod
    def parse_memory_sizes(cls, v: dict[str, Any]) -> dict[str, int]:
        return {k: parse_size(val) for k, val in v.items()}

    def total_size(self) -> int:
        return sum(self.memory.values())


class DebugConfig(BaseModel):
    slow_threshold: float = 0


default_debug_config = DebugConfig()


class Config(BaseModel):
    aramaki: Optional["AramakiConfig"] = None
    auth: "AuthConfig"
    server: "ServerConfig"
    backend: list["BackendConfig"]
    models: list["ModelInstanceConfig"]
    logging: "LoggingConfig"
    debug: DebugConfig = default_debug_config


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    directory: Path = Path(".")


class AuthConfig(BaseModel):
    static_tokens: list[str] = []


class AramakiConfig(BaseModel):
    url: str
    state_directory: Path
    secret_salt: str
    principal: str


class BackendConfig(BaseModel):
    type: str
    url: str


class LoggingConfig(BaseModel):
    log_level: str = "INFO"
    access_log_path: Path = Path("/var/log/skvaider/access.log")
