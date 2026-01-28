import asyncio
import http.server
import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from skvaider.inference.config import ModelConfig, ModelFile
from skvaider.inference.manager import Model


class ServerState:
    status = 200
    body: dict[str, Any] | None = None
    last_request_json: dict[str, Any] | None = None


@pytest.fixture
def fake_llama_server() -> Generator[tuple[str, ServerState], None, None]:
    state = ServerState()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            # Used by _wait_for_startup
            if self.path == "/health":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")
            else:
                self.send_response(404)

        def do_POST(self):
            # Used by _monitor_health
            if self.path == "/v1/completions" or self.path == "/v1/embeddings":
                length = int(self.headers.get("content-length", 0))
                if length > 0:
                    try:
                        state.last_request_json = json.loads(
                            self.rfile.read(length)
                        )
                    except Exception:
                        pass

                self.send_response(state.status)
                self.end_headers()
                if state.body is not None:
                    self.wfile.write(json.dumps(state.body).encode())
                else:
                    self.wfile.write(b"{}")
            else:
                self.send_response(404)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", state
    server.shutdown()


@asynccontextmanager
async def mock_llama_subprocess(port: int) -> AsyncGenerator[MagicMock, None]:
    """Mocks the llama-server subprocess and feeds the startup log line."""
    mock_proc = MagicMock()
    mock_proc.returncode = None

    fake_stderr = asyncio.StreamReader()
    fake_stderr.feed_data(
        f"main: HTTP server is listening, hostname: 127.0.0.1, port: {port}\n".encode()
    )
    fake_stderr.feed_eof()

    fake_stdout = asyncio.StreamReader()
    fake_stdout.feed_eof()

    mock_proc.stderr = fake_stderr
    mock_proc.stdout = fake_stdout

    async def wait():
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass

    mock_proc.wait = AsyncMock(side_effect=wait)
    mock_proc.terminate = MagicMock()
    mock_proc.kill = MagicMock()

    with patch(
        "asyncio.create_subprocess_exec", new=AsyncMock(return_value=mock_proc)
    ):
        yield mock_proc


async def await_health(model: Model, expect_healthy: bool) -> None:
    for _ in range(50):
        if model.is_healthy == expect_healthy:
            return
        await asyncio.sleep(0.01)
    assert (
        model.is_healthy == expect_healthy
    ), f"Timeout waiting for health={expect_healthy}"


@pytest.mark.parametrize(
    "model_kwargs",
    [
        {"id": "test", "files": [ModelFile(url="u", hash="h")]},
        {
            "id": "test-embed",
            "cmd_args": ["--embeddings"],
            "files": [ModelFile(url="u", hash="h")],
        },
    ],
)
async def test_health_check(
    fake_llama_server: tuple[str, ServerState],
    tmp_path: Path,
    model_kwargs: dict[str, Any],
):
    url, state = fake_llama_server
    port = int(url.split(":")[-1])

    config = ModelConfig(**model_kwargs)
    model = Model(config)
    model.datadir = tmp_path

    # Configure fast health checks
    model.health_check_interval = 0.01
    model.health_check_timeout = 0.01

    async with mock_llama_subprocess(port):
        await model.start()
        try:
            assert model.endpoint == url

            # 1. Verify it becomes healthy
            await await_health(model, True)

            # 2. Simulate failure
            state.status = 500
            await await_health(model, False)

            # 3. Simulate recovery
            state.status = 200
            await await_health(model, True)

        finally:
            await model.terminate()


async def test_health_check_embeddings(
    fake_llama_server: tuple[str, ServerState],
    tmp_path: Path,
):
    url, state = fake_llama_server
    port = int(url.split(":")[-1])

    expected_embedding = [0.1, 0.2, 0.3]
    state.body = {
        "model": "test-embed",
        "object": "list",
        "data": [
            {"embedding": expected_embedding, "index": 0, "object": "embedding"}
        ],
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }

    config = ModelConfig(
        id="test-embed",
        cmd_args=["--embeddings"],
        files=[ModelFile(url="u", hash="h")],
    )
    model = Model(config)
    model.datadir = tmp_path
    model.verification_data = {"test input": expected_embedding}

    # Configure fast health checks
    model.health_check_interval = 0.01
    model.health_check_timeout = 0.01

    async with mock_llama_subprocess(port):
        await model.start()
        try:
            # 1. Verify it sends correct input and becomes healthy
            await await_health(model, True)
            assert state.last_request_json is not None
            assert state.last_request_json["input"] == "test input"

            # 2. Simulate wrong embedding -> unhealthy
            state.body["data"][0]["embedding"] = [0.9, 0.9, 0.9]
            await await_health(model, False)

            # 3. Correct again -> healthy
            state.body["data"][0]["embedding"] = expected_embedding
            await await_health(model, True)

        finally:
            await model.terminate()
