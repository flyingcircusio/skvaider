from pathlib import Path
import svcs
from fastapi import FastAPI, Security

import skvaider.routers.admin
import skvaider.routers.openai
from skvaider.routers.openai import ModelDB
from skvaider.auth import verify_token
from skvaider.db import sessionmanager


@svcs.fastapi.lifespan
async def lifespan(app: FastAPI, registry: svcs.Registry):
    model_db = ModelDB.from_config_file(Path("models.toml"))
    # backends =

    registry.register_factory(ModelDB, lambda: model_db)
    yield {}
    if sessionmanager._engine is not None:
        # Close the DB connection
        await sessionmanager.close()


def app_factory():
    app = FastAPI(lifespan=lifespan)
    app.include_router(
        skvaider.routers.openai.router,
        prefix="/openai",
        dependencies=[Security(verify_token)],
    )
    app.include_router(
        skvaider.routers.openai.router,
        prefix="/openai",
    )

    return app
