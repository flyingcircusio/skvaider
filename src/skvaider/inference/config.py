from pathlib import Path

from pydantic import BaseModel

from skvaider.config import LoggingConfig as BaseLoggingConfig


class Config(BaseModel):
    models_dir: Path
    logging: "LoggingConfig"
    openai: "OpenAIConfig"


class OpenAIConfig(BaseModel):
    models: dict[str, "ModelConfig"]


class LoggingConfig(BaseLoggingConfig):
    access_log_path: Path = Path("/var/log/skvaider/inference-access.log")


class ModelConfig(BaseModel):
    id: str | None = None
    files: list["ModelFile"]
    cmd_args: list[str] = []
    context_size: int | None = None
    llama_server: Path = Path("llama-server")


class ModelFile(BaseModel):
    url: str
    hash: str
