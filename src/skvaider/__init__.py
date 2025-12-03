import asyncio
import os
import tomllib
from logging import getLogger
from logging.config import dictConfig

import structlog.dev
import structlog.stdlib
import svcs
from fastapi import FastAPI, Request, Security
from fastapi.responses import JSONResponse

import skvaider.routers.openai
from aramaki import Manager as AramakiManager
from skvaider.auth import verify_token
from skvaider.config import Config
from skvaider.logging import LoggingMiddleware, logging_config

log = structlog.stdlib.get_logger()


def global_exception_handler(loop, context):
    """
    This is the global handler for all unhandled exceptions in asyncio.
    """
    # context is a dictionary containing details about the exception.
    # The 'exception' key holds the exception object.
    # The 'message' key holds the error message.
    # The 'task' key (if available) holds the task that raised the exception.
    exception = context.get("exception")

    if exception:
        msg = f"Unhandled Asyncio Exception: {context.get('message', '')}\n"
        msg += f"Exception: {type(exception).__name__}: {exception}\n"

        task = context.get("task")
        if task:
            msg += f"Task: {task.get_name()} ({task})\n"

        # You can also log the 'future' or 'handle' if they exist
        future = context.get("future")
        if future:
            msg += f"Future: {future}\n"

        log.error(msg, exc_info=exception)
    else:
        # This might be a simpler message
        log.error(
            f"Unhandled Asyncio error (no exception object): {context['message']}"
        )


@svcs.fastapi.lifespan
async def lifespan(app: FastAPI, registry: svcs.Registry):
    config_file = os.environ.get("SKVAIDER_CONFIG_FILE", "config.toml")
    with open(config_file, "rb") as f:
        config_data = tomllib.load(f)
    config = Config.model_validate(config_data)

    dictConfig(logging_config(config))

    cr = structlog.dev.ConsoleRenderer.get_active()
    cr.exception_formatter = structlog.dev.plain_traceback

    loop = asyncio.get_running_loop()
    loop.set_exception_handler(global_exception_handler)

    model_config = skvaider.routers.openai.ModelConfig(config.openai.models)

    pool = skvaider.routers.openai.Pool()
    for backend_config in config.backend:
        if backend_config.type == "skvaider":
            pool.add_backend(
                skvaider.routers.openai.SkvaiderBackend(
                    backend_config.url, model_config
                )
            )
        elif backend_config.type == "ollama":
            url = backend_config.url
            if not url.startswith("http"):
                url = f"http://{url}"
            pool.add_backend(
                skvaider.routers.openai.OllamaBackend(url, model_config)
            )
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

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """
        This catches all unhandled 500 errors anywhere in the app.
        """
        log.error(
            f"Unhandled exception for request: {request.method} {request.url}",
            exc_info=exc,
        )

        return JSONResponse(
            status_code=500,
            content={"detail": "An internal server error occurred."},
        )

    return app
