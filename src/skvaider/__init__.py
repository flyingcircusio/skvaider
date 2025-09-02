import asyncio
import os
import tomllib

import svcs
from fastapi import FastAPI, Security

import skvaider.routers.openai
from aramaki import Manager as AramakiManager
from skvaider.auth import verify_token
from skvaider.config import Config


@svcs.fastapi.lifespan
async def lifespan(app: FastAPI, registry: svcs.Registry):
    config_file = os.environ.get("SKVAIDER_CONFIG_FILE", "config.toml")
    with open(config_file, "rb") as f:
        config_data = tomllib.load(f)
    config = Config.model_validate(config_data)

    pool = skvaider.routers.openai.Pool()
    for backend_config in config.backend:
        if backend_config.type != "openai":
            continue
        pool.add_backend(skvaider.routers.openai.Backend(backend_config.url))
    registry.register_value(skvaider.routers.openai.Pool, pool)

    tasks = []

    aramaki = AramakiManager(
        config.aramaki.principal,
        "skvaider",
        config.aramaki.url,
        config.aramaki.secret_salt,
        config.aramaki.state_directory,
    )
    auth_tokens = aramaki.register_collection(skvaider.auth.AuthTokens)
    registry.register_factory(
        skvaider.auth.AuthTokens, auth_tokens.get_collection_with_session
    )
    tasks.append(asyncio.create_task(aramaki.run()))

    yield {}
    # XXX cancel the collection sync tasks, maybe rewrap the aramaki task and handle the collection tasks internally
    for task in tasks:
        task.cancel()
    pool.close()


def app_factory(lifespan=lifespan):
    app = FastAPI(lifespan=lifespan)
    app.include_router(
        skvaider.routers.openai.router,
        prefix="/openai",
        dependencies=[Security(verify_token)],
    )
    return app
