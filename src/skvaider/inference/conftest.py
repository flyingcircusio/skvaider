import asyncio
import http.server
import json
import shutil
import threading
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import svcs
import svcs.fastapi
from fastapi import FastAPI
from fastapi.testclient import TestClient

from skvaider.inference import app_factory
from skvaider.inference.config import LlamaModelFile, LlamaServerModelConfig
from skvaider.inference.manager import Manager
from skvaider.inference.model import LlamaModel


class ServerState:
    status = 200
    body: dict[str, Any] | None = None
    last_request_json: dict[str, Any] | None = None


@pytest.fixture
def fake_llama_server() -> Generator[tuple[str, ServerState, int], None, None]:
    """In-process HTTP server mimicking the llama-server /health and inference endpoints.

    Yields (url, state, port). Mutate `state` to control response behaviour.

    """
    state = ServerState()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path in ("/v1/completions", "/v1/embeddings"):
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
                self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", state, port
    finally:
        shutdown_thread = threading.Thread(target=server.shutdown)
        shutdown_thread.start()
        shutdown_thread.join(timeout=1.0)


@asynccontextmanager
async def mock_llama_subprocess(
    port: int,
) -> AsyncGenerator[MagicMock, None]:
    """Patch asyncio.create_subprocess_exec and emit a fake startup log line.

    Context manager (not a fixture) so the patch scope around model.start() is
    explicit. Takes port as a parameter to stay decoupled from fake_llama_server.
    """
    mock_proc = MagicMock()
    mock_proc.returncode = None

    fake_stderr = asyncio.StreamReader()
    fake_stderr.feed_data(
        f"main: server is listening on http://127.0.0.1:{port}\n".encode()
    )
    fake_stderr.feed_data(b"main: model loaded\n")
    fake_stderr.feed_eof()

    fake_stdout = asyncio.StreamReader()
    fake_stdout.feed_eof()

    mock_proc.stderr = fake_stderr
    mock_proc.stdout = fake_stdout

    async def _wait() -> None:
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass

    mock_proc.wait = AsyncMock(side_effect=_wait)
    mock_proc.terminate = MagicMock()
    mock_proc.kill = MagicMock()

    with patch(
        "asyncio.create_subprocess_exec", new=AsyncMock(return_value=mock_proc)
    ):
        yield mock_proc


@pytest.fixture
def client(
    manager: Manager, gemma: LlamaModel
) -> Generator[TestClient, None, None]:
    @svcs.fastapi.lifespan
    async def test_lifespan(
        app: FastAPI, registry: svcs.Registry
    ) -> AsyncGenerator[None, None]:
        registry.register_value(  # pyright: ignore[reportUnknownMemberType]
            Manager, manager
        )

        yield

    with TestClient(app_factory(lifespan=test_lifespan)) as client:
        yield client


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    p = tmp_path / "models"
    p.mkdir()
    return p


@pytest.fixture
def models_cache() -> Path:
    # This is on purpose not in a tmp_path as we want to cache this
    # over multiple runs.
    cache_dir = Path(".models").absolute()
    if not cache_dir.exists():
        cache_dir.mkdir()
    return cache_dir


@pytest.fixture
async def manager(model_path: Path) -> AsyncGenerator[Manager, None]:
    m = Manager(model_path)
    yield m
    await m.shutdown()


USED_PORTS: set[int] = set()


def get_port() -> int:
    next = max(USED_PORTS, default=8000)
    USED_PORTS.add(next)
    return next


async def prepare_model(
    id: str,
    context: int,
    args: list[str],
    file: LlamaModelFile,
    models_cache: Path,
    manager: Manager,
) -> LlamaModel:
    config = LlamaServerModelConfig(
        id=id,
        files=[file],
        context_size=context,
        cmd_args=args,
        port=get_port(),
    )

    model = LlamaModel(config)
    manager.add_model(model)

    cache_dir = models_cache / model.slug
    if cache_dir.exists() and cache_dir.is_file():
        cache_dir.unlink()
    if not cache_dir.exists():
        await model.download()
        cache_dir.mkdir()
        for f in model.model_files:
            # f.rename(cache_dir / f.name)
            shutil.move(
                f, cache_dir / f.name
            )  # cannot move accross filesystems, for example in github actions runner
    # We had data in the cache. The download method is unaware of the test-fixture
    # caching. Maybe there could be a real world use case to make download() smarter.
    # The test harness needs to ensure that we restore the data from the cache as expected.
    for f in model.model_files:
        f.symlink_to(cache_dir / f.name)
    model.integrity_marker_file.touch()

    return model


@pytest.fixture
async def gemma(models_cache: Path, manager: Manager) -> LlamaModel:
    return await prepare_model(
        "gemma",
        4096,
        [],
        LlamaModelFile(
            url="https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/c90975dbd40c0c7b275fefaae758c3415c906238/gemma-3-270m-it-UD-Q4_K_XL.gguf?download=true",
            hash="e5420636e0cbfee24051ff22e9719380a3a93207a472edb18dd0c89a95f6ef80",
        ),
        models_cache,
        manager,
    )


@pytest.fixture
async def embeddinggemma(models_cache: Path, manager: Manager) -> LlamaModel:
    return await prepare_model(
        "embeddinggemma",
        4096,
        ["--embeddings", "-ngl", "0"],
        LlamaModelFile(
            url="https://huggingface.co/unsloth/embeddinggemma-300m-GGUF/resolve/main/embeddinggemma-300M-F32.gguf",
            hash="a3125072128fc76d1c1d8d19f7b095c7e3bfbf00594dcf8a8bd3bcb334935d57",
        ),
        models_cache,
        manager,
    )


@pytest.fixture
def gguf_http_server() -> Generator[str, None, None]:
    """Serve static GGUF fixture files via HTTP."""
    server_address = ("localhost", 0)  # let the OS pick an available port
    MY_FILE_DIR = Path(__file__).parent.resolve()

    class CustomHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(
                *args,
                directory=str(
                    MY_FILE_DIR
                    / "tests"
                    / "fixtures"
                    / "gguf_http_server_files"
                ),
                **kwargs,
            )

    httpd = http.server.HTTPServer(server_address, CustomHandler)
    port = httpd.server_address[1]

    thread = threading.Thread(target=httpd.serve_forever)
    thread.daemon = True
    thread.start()

    yield f"http://localhost:{port}"

    httpd.shutdown()
    thread.join()
