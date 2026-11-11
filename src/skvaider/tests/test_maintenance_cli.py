import http.server
import json
import socket
import threading
from collections.abc import Generator
from typing import Any

import pytest

from skvaider.maintenance_cli import (
    EXIT_FAILURE,
    EXIT_OK,
    EXIT_TEMPFAIL,
    main,
    maintenance_url,
)


class State:
    def __init__(self, mode: str):
        self.mode = mode
        self.posts = 0
        self.gets = 0
        self.deletes = 0
        self.last_reason = ""


class Server:
    """Mock of the inference server's /manager/maintenance endpoint."""

    def __init__(self, mode: str = "normal"):
        self.state = State(mode)
        handler = type("Handler", (_Handler,), {"mock": self})
        self.httpd = http.server.HTTPServer(("127.0.0.1", 0), handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        self.httpd.shutdown()
        self.thread.join()

    # -- behaviour ---------------------------------------------------------

    def post(self) -> tuple[int, dict[str, Any]]:
        self.state.posts += 1
        if self.state.mode == "forbid":
            return 403, {"detail": "local only"}
        return 200, {"phase": "requested"}

    def delete(self) -> tuple[int, dict[str, Any]]:
        self.state.deletes += 1
        if self.state.mode == "delete500":
            return 500, {"detail": "boom"}
        self.state.mode = "left"
        return 200, {"phase": "none"}

    def get(self) -> tuple[int, dict[str, Any]]:
        self.state.gets += 1
        mode = self.state.mode
        if mode == "normal":
            phase = ["requested", "draining", "drained"][
                min(self.state.gets - 1, 2)
            ]
        elif mode == "stuck":
            phase = "requested"
        elif mode == "lose":
            phase = "none" if self.state.posts < 2 else "drained"
        elif mode == "delete500":
            phase = "drained"
        else:
            phase = "none"
        return 200, {
            "phase": phase,
            "since": None,
            "reason": None,
            "manifest": [],
            "pending_models": [],
        }


class _Handler(http.server.BaseHTTPRequestHandler):
    mock: Server

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _send(self, code: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.mock.state.last_reason = body.get("reason", "")
        self._send(*self.mock.post())

    def do_DELETE(self) -> None:
        self._send(*self.mock.delete())

    def do_GET(self) -> None:
        self._send(*self.mock.get())


@pytest.fixture
def server() -> Generator[Server, None, None]:
    mock = Server()
    try:
        yield mock
    finally:
        mock.stop()


def enter(server: Server, *extra: str) -> int:
    return main(
        [
            "enter",
            "--url",
            server.base_url,
            "--wait",
            "2",
            "--poll-interval",
            "0.05",
            *extra,
        ]
    )


def test_maintenance_url_is_built_once():
    assert (
        maintenance_url("http://127.0.0.1:8000")
        == "http://127.0.0.1:8000/manager/maintenance"
    )
    assert (
        maintenance_url("http://127.0.0.1:8000/manager/maintenance")
        == "http://127.0.0.1:8000/manager/maintenance"
    )
    assert (
        maintenance_url("http://127.0.0.1:8000/")
        == "http://127.0.0.1:8000/manager/maintenance"
    )


def test_enter_waits_for_drained(server: Server):
    assert enter(server) == EXIT_OK
    assert server.state.posts == 1
    assert server.state.last_reason.startswith("fc-agent maintenance on ")


def test_enter_postpones_when_models_stay(server: Server):
    server.state.mode = "stuck"
    assert enter(server) == EXIT_TEMPFAIL
    assert server.state.gets >= 1


def test_enter_requests_again_after_server_restart(server: Server):
    server.state.mode = "lose"
    assert enter(server) == EXIT_OK
    assert server.state.posts >= 2


def test_enter_postpones_when_request_is_rejected(server: Server):
    server.state.mode = "forbid"
    assert enter(server) == EXIT_TEMPFAIL


def test_enter_without_inference_server_continues():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert main(["enter", "--url", f"http://127.0.0.1:{port}"]) == EXIT_OK


def test_leave_clears_maintenance(server: Server):
    assert main(["leave", "--url", server.base_url]) == EXIT_OK
    assert server.state.deletes == 1


def test_leave_fails_on_api_error(server: Server):
    server.state.mode = "delete500"
    assert main(["leave", "--url", server.base_url]) == EXIT_FAILURE


def test_leave_without_inference_server_succeeds():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert main(["leave", "--url", f"http://127.0.0.1:{port}"]) == EXIT_OK


def test_usage_error():
    with pytest.raises(SystemExit) as e:
        main([])
    assert e.value.code == 2
