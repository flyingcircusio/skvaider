import json
import re
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

    if body_json := data.get("body_json"):
        body_text = json.dumps(body_json, indent=2)
    else:
        body_text = data.get("body", b"").decode("utf-8", errors="replace")

    return f"{meta}\n\n{request_line}\n{header_lines}\n\n{body_text}"


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

    if body_json := data.get("body_json"):
        body_text = json.dumps(body_json, indent=2)
    else:
        body_text = data.get("body", b"").decode("utf-8", errors="replace")

    return f"{meta}\n\n{status_line}\n{header_lines}\n\n{body_text}"


class DebuggingMiddleware:
    """Provide debugging helpers for individual requests.

    Note: This only works together as an INNER middleware combined with the LoggingMiddleware.

    """

    directory: Path
    slow_threshold: float

    def __init__(self, app: ASGIApp, *, directory: Path, slow_threshold: float):
        self.app = app
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

        self.slow_threshold = slow_threshold

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request = Request(scope, receive)
        recorder = request.state.debug_recorder = DebugRecorder(
            request,
            self.directory,
            receive,
            send,
            self.slow_threshold,
        )
        result = await self.app(
            scope, recorder.capture_request, recorder.capture_response
        )
        await recorder.record()
        return result


class DebugRecorder:
    # XXX async file io!
    temp_file: Path | None
    triggers: list[str]
    time_start: float
    enabled: bool = False
    debug_id: str = ""

    MAX_REQUEST_BUFFER = 500 * 1024  # 500k max buffer for requests/responses

    def __init__(
        self,
        request: Request,
        directory: Path,
        receive: Receive,
        send: Send,
        slow_threshold: float = 0,
    ):
        self._orig_receive = receive
        self._orig_send = send

        self.triggers = []
        self.directory = directory
        self.slow_threshold = slow_threshold
        self.request = request
        self.temp_file = None
        self.captured_request_body = b""

        self.time_start = time.time()

        self.debug_id = sanitize_debug_id(
            request.headers.get("x-skvaider-debug-id", "")
        )
        debug_header = request.headers.get("x-skvaider-debug")
        if self.debug_id or debug_header:
            self.enabled = True
            self.triggers.append("header")

        self.captured_response_body = b""

    async def capture_request(self) -> Message:
        message = await self._orig_receive()
        if message["type"] == "http.request":
            chunk = message.get("body", b"")
            remaining_buffer_size = self.MAX_REQUEST_BUFFER - len(
                self.captured_request_body
            )
            self.captured_request_body += chunk[:remaining_buffer_size]

        return message

    async def capture_response(self, message: Message):
        if message["type"] == "http.response.start":
            self.status_code = message["status"]
            self.response_headers = {
                k.decode(): v.decode() for k, v in message.get("headers", [])
            }
        elif message["type"] == "http.response.body":
            chunk = message.get("body", b"")
            remaining_buffer_size = self.MAX_REQUEST_BUFFER - len(
                self.captured_response_body
            )
            self.captured_response_body += chunk[:remaining_buffer_size]
        await self._orig_send(message)

    def trigger_flag(self):
        if not self.triggers:
            return "-"
        _trigger_flag = {"header": "H", "slow": "S", "error": "E"}
        return _trigger_flag.get(self.triggers[0], "?")

    async def write_request(self, stem: str, data: dict[str, Any]) -> None:
        path = self.directory / f"{stem}.request"
        path.write_text(_format_request(data))

    async def write_response(self, stem: str, data: dict[str, Any]) -> None:
        path = self.directory / f"{stem}.response"
        path.write_text(_format_response(data))

    async def record(self) -> None:
        try:
            self.request_body_json = json.loads(
                self.captured_request_body.decode("utf-8")
            )
        except Exception:
            self.request_body_json = None

        try:
            self.response_body_json = json.loads(
                self.captured_response_body.decode("utf-8")
            )
        except Exception:
            self.response_body_json = None

        state = self.request.state
        request_id = state.request_id

        stem = make_stem(request_id, self.debug_id)

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
            "debug_id": self.debug_id,
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
            "body": self.captured_request_body,
            "body_json": self.request_body_json,
        }

        response_data: dict[str, Any] = {
            "request_id": request_id,
            "status_code": self.status_code,
            "headers": self.response_headers,
            "streaming": state.stream,
            "body": self.captured_response_body,
            "body_json": self.response_body_json,
            "timings": {
                "queue": round(state.time_queue, 3),
                "server": round(state.time_server, 3),
            },
        }

        await self.write_request(stem, request_data)
        await self.write_response(stem, response_data)
