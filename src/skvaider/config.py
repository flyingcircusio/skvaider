import json

from pydantic import BaseModel


class Config(BaseModel):
    database: "DatabaseConfig"
    aramaki: "AramakiConfig"
    backend: list["BackendConfig"]


class DatabaseConfig(BaseModel):
    url: str


class AramakiConfig(BaseModel):
    url: str
    enc_json_path: str = "/etc/nixos/enc.json"

    def get_aramaki_secret(self):
        enc = json.load(open(self.enc_json_path))
        return enc["parameters"]["secret_salt"]



class BackendConfig(BaseModel):
    type: str
    url: str