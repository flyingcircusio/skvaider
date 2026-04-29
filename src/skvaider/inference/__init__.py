import argparse
import asyncio
import os
import tomllib
from collections.abc import AsyncGenerator
from logging import getLogger
from logging.config import dictConfig
from typing import Any, Awaitable

import structlog
import structlog.dev
import structlog.stdlib
import svcs
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import skvaider.inference.routers.manager
import skvaider.inference.routers.metrics
import skvaider.inference.routers.models
from skvaider import global_exception_handler
from skvaider.inference.config import Config
from skvaider.inference.manager import Manager
from skvaider.logging import LoggingMiddleware, logging_config
from skvaider.utils import TaskManager

from .config import (
    LlamaServerModelConfig,
    ModelConfig,
    SystemdDockerModelConfig,
    SystemdModelConfig,
    VllmModelConfig,
)
from .model import LlamaModel, SystemdDockerModel, SystemdModel, VllmModel

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
    args, _ = parser.parse_known_args()
    with open(args.config_path, "rb") as f:
        config_data = tomllib.load(f)
    return Config.model_validate(config_data)


@svcs.fastapi.lifespan
async def lifespan(
    app: FastAPI, registry: svcs.Registry
) -> AsyncGenerator[None]:
    config = app.state.config

    verification_data = {}
    if config.embedding_verification_file:
        with open(config.embedding_verification_file, "rb") as f:
            if config.embedding_verification_file.suffix == ".json":
                import json

                verification_data = json.load(f)
            else:
                verification_data = tomllib.load(f)

    log.info("Inference manager starting...")

    try:
        config.models_dir.mkdir(exist_ok=True)
    except OSError as e:
        log.error("Failed to create models directory", error=str(e))
        raise

    # The rust-based libraries can only be configured using an OS variable ... :
    os.environ["HF_HOME"] = str(config.models_dir / ".hf")

    manager = Manager(
        models_dir=config.models_dir, log_dir=config.logging.log_dir
    )
    registry.register_value(  # pyright: ignore[reportUnknownMemberType]
        Manager, manager
    )

    seen_ports: dict[int, ModelConfig] = {}
    model_downloads: list[Awaitable[Any]] = []
    for model_config in config.openai.models:
        if duplicate_port_model := seen_ports.get(model_config.port):
            raise ValueError(
                f"{model_config.id}: port {model_config.port} already in use by {duplicate_port_model.id}."
            )
        if isinstance(model_config, LlamaServerModelConfig):
            model = LlamaModel(model_config)
        elif isinstance(model_config, VllmModelConfig):
            model = VllmModel(model_config)
        elif isinstance(model_config, SystemdDockerModelConfig):
            model = SystemdDockerModel(model_config)
        elif isinstance(model_config, SystemdModelConfig):  # pyright: ignore[reportUnnecessaryIsInstance]
            model = SystemdModel(model_config)
        else:
            raise ValueError(f"Unhandled model config: {model_config}")

        if model_config.id and model_config.id in verification_data:
            model.verification_data = verification_data[model_config.id]
        manager.add_model(model)
        model_downloads.append(model.download())

    log.info("Waiting for model downloads to finish")
    # XXX should we continue if models fail downloading?
    # otherwise if just one model fails consistently we'd loose
    # all inference servers during a reboot or so.
    await asyncio.gather(*model_downloads)

    log.info("Ready to handle requests.")

    def purge_outdated_models() -> None:
        # Our previous approach killed too much data and wasn't compatible
        # with how hugging face caches downloads.
        return

    tasks = TaskManager()
    tasks.create(asyncio.to_thread, args=(purge_outdated_models,))

    loop = asyncio.get_running_loop()
    loop.set_exception_handler(global_exception_handler)
    yield

    log.info("Shutting down...")
    await manager.shutdown()

    tasks.terminate()


def app_factory(config: Config, lifespan: Any = lifespan) -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.state.config = config
    app.include_router(skvaider.inference.routers.models.router)
    app.include_router(skvaider.inference.routers.manager.router)
    app.include_router(skvaider.inference.routers.metrics.router)

    app.add_middleware(
        LoggingMiddleware,
        logger=getLogger("skvaider.accesslog"),
        trust_remote_request_id=True,
        has_debugger=False,
        skip_paths=frozenset({"/metrics", "/manager/health"}),
    )

    @app.exception_handler(Exception)
    async def _exception_handler(  # pyright: ignore[reportUnusedFunction]
        request: Request, exc: Exception
    ) -> JSONResponse:
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


def main():
    config = load_config()

    cr = structlog.dev.ConsoleRenderer.get_active()
    cr.exception_formatter = structlog.dev.plain_traceback
    dictConfig(
        logging_config(config.logging),
    )

    app = app_factory(config, lifespan)
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        access_log=False,
    )
