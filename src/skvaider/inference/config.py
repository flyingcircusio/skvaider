from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from skvaider.config import LoggingConfig as BaseLoggingConfig


class Config(BaseModel):
    models_dir: Path
    server: "ServerConfig"
    logging: "LoggingConfig"
    openai: "OpenAIConfig"
    embedding_verification_file: Path | None = None


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class OpenAIConfig(BaseModel):
    models: list["AnyModelConfig"]


class LoggingConfig(BaseLoggingConfig):
    access_log_path: Path = Path("/var/log/skvaider/inference-access.log")
    # Directory for per-model subprocess log files (inference-<id>.log).
    # When set, each model process writes its stdout/stderr here instead of
    # being forwarded through the Python logger.
    log_dir: Path | None = None


class LlamaModelFile(BaseModel):
    url: str
    hash: str


class ModelConfig(BaseModel):
    """Shared fields for all inference backend model configs."""

    id: str
    task: Literal["chat", "embedding"]
    max_requests: int
    port: int

    @field_validator("id")
    @classmethod
    def lower_id(cls, v: str) -> str:
        return v.lower()


class LlamaServerModelConfig(ModelConfig):
    engine: Literal["llama-server"] = "llama-server"
    llama_server: Path = Path("llama-server")
    max_requests: int = 16
    files: list[LlamaModelFile]
    cmd_args: list[str] = []
    context_size: int


class VllmModelConfig(ModelConfig):
    engine: Literal["vllm"] = "vllm"
    vllm: Path = Path("vllm")
    max_requests: int = 16
    revision: str
    repo: str
    cmd_args: list[str] = []
    env: dict[str, str] = {}
    context_size: int


class SystemdModelConfig(ModelConfig):
    engine: Literal["systemd"] = "systemd"
    unit: str
    max_requests: int = 16


AnyModelConfig = Annotated[
    LlamaServerModelConfig | VllmModelConfig | SystemdModelConfig,
    Field(discriminator="engine"),
]
