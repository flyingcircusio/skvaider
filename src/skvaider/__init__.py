from pathlib import Path

import svcs
from fastapi import Depends, FastAPI

import skvaider.routers.openai
from skvaider.auth import BearerToken
from skvaider.routers.openai import ModelDB


@svcs.fastapi.lifespan
async def lifespan(app: FastAPI, registry: svcs.Registry):
    model_db = ModelDB.from_config_file(Path("models.toml"))
    # backends =

    registry.register_factory(ModelDB, lambda: model_db)
    yield {}


def app_factory():
    app = FastAPI(dependencies=[Depends(BearerToken)], lifespan=lifespan)
    app.include_router(
        skvaider.routers.openai.router,
        prefix="/openai",
    )
    return app
