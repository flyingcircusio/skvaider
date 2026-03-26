import json
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send


def sanitize_debug_id(value: str) -> str:
    sanitized = re.sub(r"[^a-z0-9_-]", "-", value.lower())
    sanitized = re.sub(r"-+", "-", sanitized)
    return sanitized[:32].strip("-")


def make_stem(request_id: str, debug_id: str) -> str:
    if debug_id:
        return f"{request_id}-{debug_id}"
    return request_id


def _format_request(data: dict[str, Any]) -> str:
    meta = f"# skvaider-debug/1  request_id={data['request_id']}"
    if data.get("debug_id"):
        meta += f"  debug_id={data['debug_id']}"
    meta += f"  triggers={','.join(data['triggers'])}"
    meta += f"  timestamp={data['timestamp']}"
    meta += f"  backend={data['backend']}"
    meta += f"  model={data['model']}"
    meta += f"  backend_endpoint={data['backend_endpoint']}"

    request_line = f"{data['method']} {data['proxy_url']}"

    header_lines = "\n".join(f"{k}: {v}" for k, v in data["headers"].items())

    body = (
        json.dumps(data["body"], indent=2)
        if data.get("body") is not None
        else ""
    )

    return f"{meta}\n\n{request_line}\n{header_lines}\n\n{body}"


def _format_response(data: dict[str, Any]) -> str:
    meta = (
        f"# skvaider-debug/1  request_id={data['request_id']}"
        f"  status_code={data['status_code']}"
        f"  streaming={data['streaming']}"
    )
    t = data.get("timings", {})
    if t:
        meta += (
            f"  timings=queue:{t.get('queue', 0)},server:{t.get('server', 0)}"
        )

    status_line = f"HTTP/1.1 {data['status_code']}"
    header_lines = "\n".join(
        f"{k}: {v}" for k, v in data.get("headers", {}).items()
    )

    body = data.get("body")
    if isinstance(body, dict):
        body_text = json.dumps(body, indent=2)
    elif body:
        body_text = str(body)
    else:
        body_text = ""

    return f"{meta}\n\n{status_line}\n{header_lines}\n\n{body_text}"


class DebuggingMiddleware:
    """Provide debugging helpers for individual requests.

    Note: This only works together as an INNER middleware combined with the LoggingMiddleware.

    """

    directory: Path
    slow_threshold: int

    def __init__(self, app: ASGIApp, *, directory: Path, slow_threshold: int):
        self.app = app
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

        self.slow_threshold = slow_threshold

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request = Request(scope, receive)
        request.state.debug_recorder = recorder = DebugRecorder(
            request, self.directory, self.slow_threshold
        )
        try:
            result = await self.app(
                scope,
                receive,
                lambda msg: recorder.capture_response(msg, send),
            )
            await recorder.record()
        finally:
            recorder.cleanup()
        return result


class DebugRecorder:
    # XXX async file io!
    temp_file: Path | None
    triggers: list[str]
    time_start: float
    enabled: bool = False
    debug_client_id: str = ""

    _capture_body: bool = False

    def __init__(
        self, request: Request, directory: Path, slow_threshold: int = 0
    ):
        self.triggers = []
        self.directory = directory
        self.slow_threshold = slow_threshold
        self.request = request
        self.temp_file = None

        self.time_start = time.time()

        debug_id_header = sanitize_debug_id(
            request.headers.get("x-skvaider-debug-id", "")
        )
        debug_header = request.headers.get("x-skvaider-debug")
        if debug_id_header or debug_header:
            self.enabled = True
            self.triggers.append("header")

        self.response_buffer = tempfile.NamedTemporaryFile(
            delete=False,
            dir=self.directory,
            suffix=".tmp",
        )

    async def capture_response(self, message: Message, send: Send):
        if message["type"] == "http.response.start":
            self.status_code = message["status"]
            self.response_headers = {
                k.decode(): v.decode() for k, v in message.get("headers", [])
            }
        elif message["type"] == "http.response.body" and self._capture_body:
            chunk = message.get("body", b"")
            self.response_buffer.write(chunk)
        await send(message)

    def trigger_flag(self):
        if not self.triggers:
            return "-"
        _trigger_flag = {"header": "H", "slow": "S", "error": "E"}
        return _trigger_flag.get(self.triggers[0], "?")

    def cleanup(self):
        Path(self.response_buffer.name).unlink()

    async def write_request(self, stem: str, data: dict[str, Any]) -> None:
        path = self.directory / f"{stem}.request"
        path.write_text(_format_request(data))

    async def write_response(self, stem: str, data: dict[str, Any]) -> None:
        path = self.directory / f"{stem}.response"
        path.write_text(_format_response(data))

    async def record(self) -> None:
        self.response_buffer.seek(0)

        try:
            request_data = await self.request.json()
            self.request_body = dict(request_data)
        except Exception:
            self.request_body = ""

        # XXX memory
        try:
            self.response_body = json.loads(
                self.response_buffer.read().decode("utf-8")
            )
        except Exception:
            self.response_body = self.response_buffer.read().decode("utf-8")

        state = self.request.state
        request_id = state.request_id

        stem = make_stem(request_id, self.debug_client_id)

        if self.slow_threshold and (
            time.time() - self.time_start > self.slow_threshold
        ):
            self.triggers.append("slow")
        if self.status_code >= 400:
            self.triggers.append("error")

        if not self.triggers:
            return

        backend_url = ""
        if state.backend:
            backend_url = state.backend.url

        request_data: dict[str, Any] = {
            "request_id": request_id,
            "debug_client_id": self.debug_client_id,
            "timestamp": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.000Z"
            ),
            "triggers": self.triggers,
            "method": self.request.method,
            "proxy_url": str(self.request.url),
            "headers": dict(self.request.headers),
            "backend": backend_url,
            "model": getattr(state, "model", "n/a"),
            "backend_endpoint": getattr(state, "backend_endpoint", ""),
            "body": self.request_body,
        }

        response_data: dict[str, Any] = {
            "request_id": request_id,
            "status_code": self.status_code,
            "headers": self.response_headers,
            "streaming": state.stream,
            "body": self.response_body,
            "timings": {
                "queue": round(state.time_queue, 3),
                "server": round(state.time_server, 3),
            },
        }

        await self.write_request(stem, request_data)
        await self.write_response(stem, response_data)
