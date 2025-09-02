from pathlib import Path

from pydantic import BaseModel


class Config(BaseModel):
    aramaki: "AramakiConfig"
    backend: list["BackendConfig"]


class AramakiConfig(BaseModel):
    url: str
    state_directory: Path
    secret_salt: str
    principal: str
    collection: str = "fc.directory.ai.token"


class BackendConfig(BaseModel):
    type: str
    url: str
