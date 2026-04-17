import asyncio
import http.server
import json
import shutil
import threading
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, Generator, Literal

import httpx
import pytest
import svcs
import svcs.fastapi
from fastapi import FastAPI
from fastapi.testclient import TestClient

from skvaider.inference import app_factory
from skvaider.inference.config import (
    Config,
    LlamaModelFile,
    LlamaServerModelConfig,
    LoggingConfig,
    OpenAIConfig,
    ServerConfig,
)
from skvaider.inference.manager import Manager
from skvaider.inference.model import LlamaModel
from skvaider.utils import ModelAPI


class _AsyncTestClient:
    """Adapts a sync TestClient to the async request() interface ModelAPI expects."""

    def __init__(self, client: TestClient):
        self._client = client

    async def request(self, *args: Any, **kwargs: Any) -> httpx.Response:
        kwargs.pop(
            "timeout", None
        )  # https://github.com/Kludex/starlette/issues/1108
        return await asyncio.to_thread(self._client.request, *args, **kwargs)


class TestAPI(ModelAPI):
    """ModelAPI subclass that exchanges the httpx.AsyncClient for a TestClient."""

    __test__ = False

    def __init__(self, client: TestClient):
        self.base_url = ""
        self.client = _AsyncTestClient(client)  # type: ignore[assignment]


@pytest.fixture
def test_api(client: TestClient) -> TestAPI:
    return TestAPI(client)


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

    config = Config(
        models_dir=manager.models_dir,
        server=ServerConfig(),
        logging=LoggingConfig(),
        openai=OpenAIConfig(models=[]),
    )

    with TestClient(
        app_factory(config=config, lifespan=test_lifespan)
    ) as client:
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
    cache_dir = Path("var/tests/models").absolute()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture
async def manager(
    model_path: Path, tmp_path: Path
) -> AsyncGenerator[Manager, None]:
    m = Manager(model_path, log_dir=tmp_path)
    yield m
    await m.shutdown()


USED_PORTS: set[int] = set()


def get_port() -> int:
    next = max(USED_PORTS, default=9000)
    USED_PORTS.add(next)
    return next


class OpenAIServerMock(object):
    response_status: int = 200
    response: dict[str, Any] | None = None
    last_request_json: dict[str, Any]
    host: str
    port: int

    def __call__(self, *args: Any, **kw: Any):
        handler = OpenAIServerMockHandler(self, *args, **kw)
        return handler

    @property
    def endpoint(self):
        return f"http://{self.host}:{self.port}"


class OpenAIServerMockHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, mock: OpenAIServerMock, *args: Any, **kw: Any):
        self.mock = mock
        super().__init__(*args, **kw)

    def do_POST(self):
        if self.path in ("/v1/completions", "/v1/embeddings"):
            length = int(self.headers.get("content-length", 0))
            if length > 0:
                try:
                    self.mock.last_request_json = json.loads(
                        self.rfile.read(length)
                    )
                except Exception:
                    pass
            self.send_response(self.mock.response_status)
            self.end_headers()
            if self.mock.response is not None:
                self.wfile.write(json.dumps(self.mock.response).encode())
            else:
                self.wfile.write(b"{}")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture
def openai_server() -> Generator[OpenAIServerMock, None, None]:
    """In-process HTTP server an OpenAI server.

    Yields an OpenAIServerMock instance. Mutate its attribute to change behaviour.

    """
    openai_server_mock = OpenAIServerMock()
    openai_server_mock.host = "127.0.0.1"
    server = http.server.HTTPServer(
        (openai_server_mock.host, 0), openai_server_mock
    )
    openai_server_mock.port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield openai_server_mock
    finally:
        shutdown_thread = threading.Thread(target=server.shutdown)
        shutdown_thread.start()
        shutdown_thread.join(timeout=1.0)


async def prepare_model(
    id: str,
    context: int,
    args: list[str],
    file: LlamaModelFile,
    models_cache: Path,
    manager: Manager,
    task: Literal["chat", "embedding"],
) -> LlamaModel:
    config = LlamaServerModelConfig(
        id=id,
        files=[file],
        context_size=context,
        cmd_args=args,
        port=get_port(),
        task=task,
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
        task="chat",
    )


@pytest.fixture
async def embeddinggemma(models_cache: Path, manager: Manager) -> LlamaModel:
    return await prepare_model(
        "embeddinggemma",
        4096,
        ["-ngl", "0"],
        LlamaModelFile(
            url="https://huggingface.co/unsloth/embeddinggemma-300m-GGUF/resolve/main/embeddinggemma-300M-F32.gguf",
            hash="a3125072128fc76d1c1d8d19f7b095c7e3bfbf00594dcf8a8bd3bcb334935d57",
        ),
        models_cache,
        manager,
        task="embedding",
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
