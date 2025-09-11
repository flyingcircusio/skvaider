import os
import tomllib
from logging import getLogger
from logging.config import dictConfig

import svcs
from fastapi import FastAPI, Security

import skvaider.routers.openai
from aramaki import Manager as AramakiManager
from skvaider.auth import verify_token
from skvaider.config import Config
from skvaider.logging import LoggingMiddleware, logging_config


@svcs.fastapi.lifespan
async def lifespan(app: FastAPI, registry: svcs.Registry):
    config_file = os.environ.get("SKVAIDER_CONFIG_FILE", "config.toml")
    with open(config_file, "rb") as f:
        config_data = tomllib.load(f)
    config = Config.model_validate(config_data)

    dictConfig(logging_config(config))

    pool = skvaider.routers.openai.Pool()
    for backend_config in config.backend:
        if backend_config.type != "openai":
            continue
        pool.add_backend(skvaider.routers.openai.Backend(backend_config.url))
    registry.register_value(skvaider.routers.openai.Pool, pool)

    aramaki = AramakiManager(
        config.aramaki.principal,
        "skvaider",
        config.aramaki.url,
        config.aramaki.secret_salt,
        config.aramaki.state_directory,
    )
    aramaki.start()
    auth_tokens = aramaki.register_collection(skvaider.auth.AuthTokens)
    registry.register_factory(
        skvaider.auth.AuthTokens, auth_tokens.get_collection_with_session
    )
    yield {}
    aramaki.stop()
    pool.close()


def app_factory(lifespan=lifespan):
    app = FastAPI(lifespan=lifespan)
    app.include_router(
        skvaider.routers.openai.router,
        prefix="/openai",
        dependencies=[Security(verify_token)],
    )
    app.add_middleware(
        LoggingMiddleware, logger=getLogger("skvaider.accesslog")
    )
    return app
