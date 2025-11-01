import ipaddress
import logging
import time
from typing import Awaitable, Callable

import structlog
from fastapi import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from skvaider import Config

log = structlog.stdlib.get_logger()


class LoggingMiddleware:
    def __init__(self, app: ASGIApp, *, logger: logging.Logger):
        self.app = app
        self._logger = logger

    async def inner_send(
        self, message: Message, send: Send, status_code: list[int]
    ):
        if message["type"] == "http.response.start":
            status_code[0] = message["status"]

        await send(message)

    def inner_send_factory(
        self, send: Send, status_code: list[int]
    ) -> Callable[[Message], Awaitable[None]]:
        return lambda message: self.inner_send(message, send, status_code)

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        start_time = time.perf_counter()
        client = scope["client"]
        anon_ip = "no ip"
        if client:
            try:
                client_ip = ipaddress.ip_network(client[0])
                anon_net = None
                if client_ip.version == 4:
                    anon_net = client_ip.supernet(new_prefix=24)
                if client_ip.version == 6:
                    anon_net = client_ip.supernet(new_prefix=64)
                anon_ip = str(anon_net.network_address)
            except ValueError:
                pass

        status_code = [500]
        try:
            await self.app(
                scope, receive, self.inner_send_factory(send, status_code)
            )
        finally:
            process_time = round(time.perf_counter() - start_time, 3)
            request = Request(scope, receive)

            model = "n/a"
            try:
                model = request.state.model
            except Exception:
                pass

            backend = "n/a"
            try:
                backend = request.state.backend.url
            except Exception:
                pass

            stream = "-"
            try:
                if request.state.stream:
                    stream = "S"
            except Exception:
                pass

            # Timings
            # - warmup time
            # - queuing time
            # - time to first chunk
            # - total time
            # Status codes
            # - streaming
            # response size
            # ... tokens?
            # XXX queue sizes - number of parallel requests  / with same model / on same backend / ...
            self._logger.info(
                f'{anon_ip} {model} {backend} -/-/{process_time} {status_code[0]} 0 {stream} "{request.method} {request.url.path}" '
            )


def logging_config(config: Config) -> dict:
    return {
        "version": 1,
        "disable_existing_loggers": False,
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
