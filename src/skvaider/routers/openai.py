import tomllib
import os
from pathlib import Path
from typing import Any, AsyncGenerator, Generic, TypeVar

import httpx
import svcs
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()

T = TypeVar("T")


class Backend:
    def __init__(self):
        # Use OLLAMA_HOST environment variable or default to localhost:11434
        ollama_host = os.getenv("OLLAMA_HOST", "127.0.0.1:11434")
        if not ollama_host.startswith("http"):
            ollama_host = f"http://{ollama_host}"
        self.url = ollama_host

    async def post(self, path: str, data: dict):
        async with httpx.AsyncClient() as client:
            r = await client.post(self.url + path, json=data, timeout=120)
            return r.json()
    
    async def post_stream(self, path: str, data: dict) -> AsyncGenerator[str, None]:
        """Stream responses from the backend"""
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST", 
                self.url + path, 
                json=data, 
                timeout=120
            ) as response:
                async for chunk in response.aiter_text():
                    if chunk.strip():
                        yield chunk


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
    backend = Backend()
    
    # Check if streaming is requested
    stream = request_data.get("stream", False)
    
    if stream:
        # Return streaming response
        async def generate():
            async for chunk in backend.post_stream("/v1/chat/completions", request_data):
                yield chunk
        
        return StreamingResponse(
            generate(), 
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
    else:
        # Return regular JSON response
        return await backend.post("/v1/chat/completions", request_data)


@router.post("/v1/completions")
async def completions(
    r: Request, services: svcs.fastapi.DepContainer
) -> Any:
    request_data = await r.json()
    request_data["store"] = False
    backend = Backend()
    
    # Check if streaming is requested
    stream = request_data.get("stream", False)
    
    if stream:
        # Return streaming response
        async def generate():
            async for chunk in backend.post_stream("/v1/completions", request_data):
                yield chunk
        
        return StreamingResponse(
            generate(), 
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
    else:
        # Return regular JSON response
        return await backend.post("/v1/completions", request_data)
