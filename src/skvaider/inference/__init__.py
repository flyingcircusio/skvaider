import asyncio
import logging
import os
import shutil
import tomllib
from logging import getLogger
from logging.config import dictConfig

import structlog
import structlog.dev
import structlog.stdlib
import svcs
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import skvaider.inference.manager
from skvaider import global_exception_handler
from skvaider.inference.config import Config
from skvaider.inference.manager import Manager
from skvaider.inference.routers import models
from skvaider.logging import LoggingMiddleware, logging_config

log = structlog.stdlib.get_logger()


@svcs.fastapi.lifespan
async def lifespan(app: FastAPI, registry: svcs.Registry):
    config_file = os.environ.get(
        "SKVAIDER_CONFIG_FILE", "config-inference.toml"
    )
    with open(config_file, "rb") as f:
        config_data = tomllib.load(f)
    config = Config.model_validate(config_data)

    cr = structlog.dev.ConsoleRenderer.get_active()
    cr.exception_formatter = structlog.dev.plain_traceback

    log.info("Inference manager starting...")

    try:
        config.models_dir.mkdir(exist_ok=True)
    except OSError as e:
        log.error("Failed to create models directory", error=str(e))
        raise

    manager = Manager(models_dir=config.models_dir)
    registry.register_value(Manager, manager)

    model_downloads = []
    for model_config in config.openai.models:
        model = skvaider.inference.manager.Model(model_config)
        manager.add_model(model)
        model_downloads.append(model.download())

    log.info("Waiting for model downloads to finish")
    # XXX should we continue if models fail downloading?
    # otherwise if just one model fails consistently we'd loose
    # all inference servers during a reboot or so.
    await asyncio.gather(*model_downloads)

    log.info("Ready to handle requests.")

    def purge_outdated_models():
        worklist = set(m.absolute() for m in manager.models_dir.glob("*"))
        worklist = worklist - set(
            m.datadir.absolute() for m in manager.models.values()
        )
        for dir in worklist:
            log.info(f"Removing outdated model data: {dir}")
            if not dir.is_dir():
                dir.unlink()
            else:
                shutil.rmtree(dir, ignore_errors=True)

    tasks = []
    tasks.append(asyncio.create_task(asyncio.to_thread(purge_outdated_models)))

    loop = asyncio.get_running_loop()
    loop.set_exception_handler(global_exception_handler)
    dictConfig(
        logging_config(config.logging)
    )  # XXX makes us swallow lifespan errors?

    yield
    log.info("Shutting down...")
    await manager.shutdown()

    for task in tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def app_factory(lifespan=lifespan):
    app = FastAPI(lifespan=lifespan)
    app.include_router(models.router)

    # XXX
    @app.get("/manager/health")
    async def health():
        return {"status": "ok"}

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
