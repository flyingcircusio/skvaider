from pathlib import Path

from pydantic import BaseModel


class Config(BaseModel):
    aramaki: "AramakiConfig"
    backend: list["BackendConfig"]
    logging: "LoggingConfig"


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
    access_log_path: Path = "/var/log/skvaider/access.log"
