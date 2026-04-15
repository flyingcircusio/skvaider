import argparse
import asyncio
import os
import tomllib
from asyncio import AbstractEventLoop
from collections.abc import AsyncGenerator
from logging import getLogger
from logging.config import dictConfig
from typing import Any

import starlette.requests
import structlog.dev
import structlog.stdlib
import svcs
import uvicorn
from fastapi import FastAPI, Request, Security
from fastapi.responses import JSONResponse

import skvaider.auth
import skvaider.proxy.backends
import skvaider.proxy.pool
import skvaider.routers.admin
import skvaider.routers.metrics
import skvaider.routers.openai
from aramaki import Manager as AramakiManager
from skvaider.auth import verify_token
from skvaider.config import Config
from skvaider.debug import DebuggingMiddleware
from skvaider.logging import LoggingMiddleware, logging_config

log = structlog.stdlib.get_logger()


def load_config() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        dest="config_path",
        default="config.toml",
        help="Path to the configuration file",
    )
    args = parser.parse_args()
    with open(args.config_path, "rb") as f:
        config_data = tomllib.load(f)
    return Config.model_validate(config_data)


def global_exception_handler(
    loop: AbstractEventLoop, context: dict[str, Any]
) -> None:
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
async def lifespan(
    app: FastAPI, registry: svcs.Registry
) -> AsyncGenerator[None]:
    config_file = os.environ.get("SKVAIDER_CONFIG_FILE", "config.toml")
    with open(config_file, "rb") as f:
        config_data = tomllib.load(f)
    config = Config.model_validate(config_data)

    loop = asyncio.get_running_loop()

    loop.set_debug(True)
    import logging

    logging.getLogger("asyncio").setLevel(logging.WARNING)

    loop.set_exception_handler(global_exception_handler)

    backends: list[skvaider.proxy.backends.Backend] = []
    for backend_config in config.backend:
        if backend_config.type == "skvaider":
            backends.append(
                skvaider.proxy.backends.SkvaiderBackend(backend_config.url)
            )
        else:
            raise TypeError(backend_config.type)

    pool = skvaider.proxy.pool.Pool(config.models, backends)
    registry.register_value(  # pyright: ignore[reportUnknownMemberType]
        skvaider.proxy.pool.Pool, pool
    )
    aramaki = None
    if config.aramaki:
        aramaki = AramakiManager(
            config.aramaki.principal,
            "skvaider",
            config.aramaki.url,
            config.aramaki.secret_salt,
            config.aramaki.state_directory,
        )
        aramaki.start()
        auth_tokens = aramaki.register_collection(skvaider.auth.AuthTokens)
        registry.register_factory(  # pyright: ignore[reportUnknownMemberType]
            skvaider.auth.AuthTokens,
            auth_tokens.get_collection_with_session,
            enter=False,
        )

    if config.auth.admin_tokens:
        registry.register_value(  # pyright: ignore[reportUnknownMemberType]
            skvaider.auth.StaticAuthTokens,
            skvaider.auth.StaticAuthTokens(config.auth.admin_tokens),
        )

    yield
    if aramaki:
        aramaki.stop()
    pool.close()


def app_factory(
    lifespan: Any = lifespan,
) -> FastAPI:
    config_file = os.environ.get("SKVAIDER_CONFIG_FILE", "config.toml")
    with open(config_file, "rb") as f:
        config_data = tomllib.load(f)
    config = Config.model_validate(config_data)

    app = FastAPI(lifespan=lifespan)
    app.include_router(
        skvaider.routers.openai.router,
        prefix="/openai",
        dependencies=[Security(verify_token)],
    )
    app.include_router(skvaider.routers.metrics.router)
    app.include_router(skvaider.routers.admin.router)
    app.add_middleware(
        DebuggingMiddleware,
        directory=config.server.directory / "debug",
        slow_threshold=config.debug.slow_threshold,
    )
    app.add_middleware(
        LoggingMiddleware,
        logger=getLogger("skvaider.accesslog"),
        trust_remote_request_id=False,
        has_debugger=True,
    )

    @app.exception_handler(Exception)
    async def _global_exception_handler(  # pyright: ignore[reportUnusedFunction]
        request: Request, exc: Exception
    ) -> JSONResponse:
        """
        This catches all unhandled 500 errors anywhere in the app.
        """
        if isinstance(exc, starlette.requests.ClientDisconnect):
            pass
        else:
            backend = "n/a"
            model = "n/a"
            try:
                backend = request.state.backend.url
            except Exception:
                pass
            try:
                model = request.state.model
            except Exception:
                pass

            log.error(
                f"Unhandled exception for request: {request.method} {request.url} - backend={backend} model={model}",
                exc_info=exc,
            )

        return JSONResponse(
            status_code=500,
            content={"detail": "An internal server error occurred."},
        )

    return app


def main():
    config = load_config()

    cr = structlog.dev.ConsoleRenderer.get_active()
    cr.exception_formatter = structlog.dev.plain_traceback
    dictConfig(
        logging_config(config.logging),
    )

    uvicorn.run(
        "skvaider:app_factory",
        host=config.server.host,
        port=config.server.port,
        factory=True,
    )
