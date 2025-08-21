import svcs
from fastapi import FastAPI, Security

import skvaider.routers.admin
import skvaider.routers.openai
from skvaider.auth import verify_token
from skvaider.db import DBSession, DBSessionManager


@svcs.fastapi.lifespan
async def lifespan(app: FastAPI, registry: svcs.Registry):
    DB_URL = "postgresql+psycopg://skvaider:foobar@localhost:5432/skvaider"
    sessionmanager = DBSessionManager(DB_URL)

    async def get_db_session():
        async with sessionmanager.session() as session:
            print(session)
            yield session

    registry.register_factory(DBSession, get_db_session)

    pool = skvaider.routers.openai.Pool()
    pool.add_backend(skvaider.routers.openai.Backend("http://127.0.0.1:11434"))
    registry.register_value(skvaider.routers.openai.Pool, pool)

    yield {}

    pool.close()
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
