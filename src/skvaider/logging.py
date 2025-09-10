import logging
import time
from typing import Callable

import structlog
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from skvaider import Config

log = structlog.stdlib.get_logger()


class LoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, *, logger: logging.Logger):
        self._logger = logger
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            data = await request.json()
            model = data.get("model", "n/a")
        except Exception:
            model = "n/a"
        try:
            backend = request.state.backend.url
        except AttributeError:
            backend = "n/a"

        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = time.perf_counter() - start_time

        self._logger.info(
            f"{request.method} {request.url.path} {model} -> {backend} {response.status_code} {process_time}"
        )

        return response


def logging_config(config: Config) -> dict:
    return {
        "version": 1,
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
            },
            "accesslog": {
                "class": "logging.FileHandler",
                "filename": config.logging.access_log_path,
                "formatter": "accesslog",
            },
        },
        "root": {
            "handlers": ["console"],
        },
        "formatters": {
            "accesslog": {
                "format": "%(asctime)s: %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            }
        },
        "loggers": {
            "skvaider": {
                "level": config.logging.log_level,
                "handlers": ["console"],
            },
            "aramaki": {
                "level": config.logging.log_level,
                "handlers": ["console"],
            },
            "skvaider.accesslog": {
                "level": "INFO",
                "handlers": ["accesslog"],
                "propagate": False,
            },
        },
    }
