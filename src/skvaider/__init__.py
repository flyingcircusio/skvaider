from pathlib import Path

import svcs
from fastapi import FastAPI, Security

import skvaider.routers.admin
import skvaider.routers.openai
from skvaider.auth import verify_token
from skvaider.db import DBSession, DBSessionManager
from skvaider.routers.openai import ModelDB


@svcs.fastapi.lifespan
async def lifespan(app: FastAPI, registry: svcs.Registry):
    model_db = ModelDB.from_config_file(Path("models.toml"))
    DB_URL = "postgresql+psycopg://skvaider:foobar@localhost:5432/skvaider"
    sessionmanager = DBSessionManager(DB_URL)

    async def get_db_session():
        async with sessionmanager.session() as session:
            print(session)
            yield session

    # backends =

    registry.register_factory(DBSession, get_db_session)
    registry.register_value(ModelDB, model_db)
    yield {}

    await sessionmanager.close()


def app_factory():
    app = FastAPI(lifespan=lifespan)
    app.include_router(
        skvaider.routers.openai.router,
        prefix="/openai",
        dependencies=[Security(verify_token)],
    )
    app.include_router(
        skvaider.routers.admin.router,
        prefix="/admin",
    )

    return app
