from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator

from skvaider.config import LoggingConfig as BaseLoggingConfig


class Config(BaseModel):
    models_dir: Path
    logging: "LoggingConfig"
    openai: "OpenAIConfig"
    embedding_verification_file: Path | None = None
    manager: "ManagerConfig"


class ManagerConfig(BaseModel):
    backend: Literal["cpu", "rocm"]


class OpenAIConfig(BaseModel):
    models: list["ModelConfig"]


class LoggingConfig(BaseLoggingConfig):
    access_log_path: Path = Path("/var/log/skvaider/inference-access.log")


class ModelConfig(BaseModel):
    id: str
    cmd_args: list[str] = []
    context_size: int = 0
    llama_server: Path = Path("llama-server")
    files: list["ModelFile"]

    @field_validator("id")
    @classmethod
    def lower_id(cls, v: str) -> str:
        return v.lower()


class ModelFile(BaseModel):
    url: str
    hash: str
