import json
from pathlib import Path

from pydantic import BaseModel


class Config(BaseModel):
    database: "DatabaseConfig"
    aramaki: "AramakiConfig"
    backend: list["BackendConfig"]


class DatabaseConfig(BaseModel):
    url: str


class AramakiConfig(BaseModel):
    url: str
    state_directory: Path
    enc_json_path: str = "/etc/nixos/enc.json"
    collection: str = "fc.directory.ai.token"

    @property
    def secret(self):
        enc = json.load(open(self.enc_json_path))
        return enc["parameters"]["secret_salt"]

    @property
    def principal(self):
        enc = json.load(open(self.enc_json_path))
        return enc["name"]


class BackendConfig(BaseModel):
    type: str
    url: str
