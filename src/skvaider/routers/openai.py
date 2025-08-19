import tomllib
from pathlib import Path
from typing import Any, Generic, TypeVar

import httpx
import svcs
from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()

T = TypeVar("T")


class Backend:
    url: str = "http://localhost:8001"

    async def post(self, path: str, data: dict):
        async with httpx.AsyncClient() as client:
            r = await client.post(self.url + path, json=data, timeout=120)
            return r.json()


class AIModel(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str


class ModelDB:
    models: dict[str, AIModel]

    def __init__(self):
        self.models = {}

    @staticmethod
    def from_config_file(config_file: Path) -> "ModelDB":
        db = ModelDB()
        with config_file.open("rb") as f:
            data = tomllib.load(f)
            for model in data["models"]:
                db.models[model["id"]] = AIModel(
                    id=model["id"],
                    created=model["created"],
                    owned_by=model["owned_by"],
                )
        return db


class ListResponse(BaseModel, Generic[T]):
    object: str = "list"
    data: list[T]


@router.get("/v1/models")
async def list_models(
    services: svcs.fastapi.DepContainer,
) -> ListResponse[AIModel]:
    model_db = services.get(ModelDB)
    return ListResponse[AIModel](data=model_db.models.values())


@router.get("/v1/models/{model_id}")
async def get_model(
    model_id: str, services: svcs.fastapi.DepContainer
) -> AIModel:
    model_db = services.get(ModelDB)
    return model_db.models[model_id]


@router.post("/v1/chat/completions")
async def chat_completions(
    r: Request, services: svcs.fastapi.DepContainer
) -> Any:
    request_data = await r.json()
    request_data["store"] = False
    # backend = services.get(Backends)
    backend = Backend()
    return await backend.post("/v1/chat/completions", request_data)
