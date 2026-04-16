import ipaddress
import logging
import time
from typing import Any

import shortuuid
import structlog
from fastapi import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from skvaider.config import LoggingConfig

log = structlog.stdlib.get_logger()


class LoggingMiddleware:
    trust_remote_request_id = False

    def __init__(
        self,
        app: ASGIApp,
        *,
        logger: logging.Logger,
        trust_remote_request_id: bool,
        has_debugger: bool,
        skip_paths: frozenset[str] = frozenset({"/metrics"}),
    ):
        self.app = app
        self._logger = logger
        self.trust_remote_request_id
        self.has_debugger = has_debugger
        self.skip_paths = skip_paths

    async def _capture_status(
        self, message: Message, send: Send, request: Request
    ):
        if message["type"] == "http.response.start":
            request.state.status_code = message["status"]
        await send(message)

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
                if anon_net is not None:
                    # This shouldn't happen and we could likely assert on non-None-ness,
                    # but I'm not willing to (have Skvaider) die on that hill.
                    anon_ip = str(anon_net.network_address)
            except ValueError:
                pass

        request = Request(scope, receive)
        request.state.status_code = 500

        request.state.request_id = None
        if self.trust_remote_request_id:
            request.state.request_id = request.headers.get(
                "x-skvaider-request-id"
            )
        if not request.state.request_id:
            request.state.request_id = shortuuid.uuid()[:8]
        request.state.model = "n/a"
        request.state.backend = None
        request.state.stream = False
        request.state.tokens_prompt = 0
        request.state.tokens_completion = 0
        request.state.tokens_total = 0
        request.state.time_queue = 0.0
        request.state.time_server = 0.0
        request.state.retries = 0
        request.state.parallel_total = 0
        request.state.parallel_model = 0
        request.state.parallel_backend = 0
        request.state.backend_endpoint = ""
        request.state.response_headers = {}

        try:
            await self.app(
                scope,
                receive,
                lambda msg: self._capture_status(msg, send, request),
            )
        finally:
            if request.url.path in self.skip_paths:
                return

            process_time = round(time.perf_counter() - start_time, 3)
            backend = (
                request.state.backend.url if request.state.backend else "n/a"
            )
            stream = "S" if request.state.stream else "-"

            if self.has_debugger:
                debug_flag = request.state.debug_recorder.trigger_flag()
            else:
                debug_flag = "_"

            time_queue = round(request.state.time_queue, 3)
            time_server = round(request.state.time_server, 3)
            p = request.state
            # XXX when streaming: add time to time first token
            self._logger.info(
                f'{anon_ip} {p.model} {backend} {p.request_id} {time_queue}/{time_server}/{process_time} {p.status_code} {p.tokens_prompt}/{p.tokens_completion}/{p.tokens_total} {stream}{debug_flag} {p.retries} {p.parallel_total}/{p.parallel_model}/{p.parallel_backend} "{request.method} {request.url.path}" '
            )


def logging_config(config: LoggingConfig) -> dict[str, Any]:
    if not config.log_dir.exists():
        config.log_dir.mkdir(parents=True, exist_ok=True)
    dictConfig: dict[str, Any] = {
        "version": 1,
        "handlers": {
            "console": {"class": "logging.StreamHandler", "formatter": "plain"},
            "accesslog": {
                "class": "logging.FileHandler",
                "filename": config.log_dir / "access.log",
                "formatter": "accesslog",
            },
        },
        "formatters": {
            "accesslog": {
                "format": "%(asctime)s: %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            },
            "plain": {
                "format": "%(message)s",
            },
        },
        "loggers": {
            "skvaider.accesslog": {
                "level": "INFO",
                "handlers": ["accesslog"],
                "propagate": False,
            },
        },
        "root": {"handlers": ["console"]},
    }

    common_config = {
        "level": config.log_level,
        "handlers": ["console"],
        "propagate": False,
    }

    for logger in [
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "uvicorn.lifespan",
        "fastapi",
        "skvaider",
        "aramaki",
    ]:
        dictConfig["loggers"][logger] = common_config

    return dictConfig
