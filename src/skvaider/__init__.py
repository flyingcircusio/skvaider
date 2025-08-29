import asyncio
import os
import tomllib

import svcs
from fastapi import FastAPI, Security

import skvaider.routers.openai
from skvaider.aramaki import Manager as AramakiManager
from skvaider.aramaki.collection_replication import CollectionReplicationManager
from skvaider.auth import verify_token
from skvaider.collection_replicator import AITokenReplicator
from skvaider.config import Config
from skvaider.db import DBSession, DBSessionManager


@svcs.fastapi.lifespan
async def lifespan(app: FastAPI, registry: svcs.Registry):
    config_file = os.environ.get("SKVAIDER_CONFIG_FILE", "config.toml")
    with open(config_file, "rb") as f:
        config_data = tomllib.load(f)
    config = Config.model_validate(config_data)

    sessionmanager = DBSessionManager(config.database.url)

    async def get_db_session():
        async with sessionmanager.session() as session:
            yield session

    registry.register_factory(DBSession, get_db_session)

    pool = skvaider.routers.openai.Pool()
    for backend_config in config.backend:
        if backend_config.type != "openai":
            continue
        pool.add_backend(skvaider.routers.openai.Backend(backend_config.url))
    registry.register_value(skvaider.routers.openai.Pool, pool)

    aramaki_manager = AramakiManager(
        config.aramaki.principal,
        "skvaider",
        config.aramaki.url,
        config.aramaki.secret,
    )
    ai_token_replicator = AITokenReplicator(registry)
    CollectionReplicationManager(
        aramaki_manager,
        "fc.directory.ai.token",
        config.aramaki.state_directory,
        ai_token_replicator.update,
        ai_token_replicator.start_full_sync,
        ai_token_replicator.end_full_sync,
    )
    aramaki_manager_task = asyncio.create_task(aramaki_manager.run())

    yield {}

    aramaki_manager_task.cancel()
    pool.close()
    await sessionmanager.close()


def app_factory():
    app = FastAPI(lifespan=lifespan)
    app.include_router(
        skvaider.routers.openai.router,
        prefix="/openai",
        dependencies=[Security(verify_token)],
    )

    return app
