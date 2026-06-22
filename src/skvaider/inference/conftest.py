import asyncio
import http.server
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

from skvaider.dummy_engine import DummyModel
from skvaider.inference import app_factory
from skvaider.inference.config import (
    Config,
    DummyModelConfig,
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
    manager: Manager, gemma: DummyModel
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


@pytest.fixture
async def openai_server() -> AsyncGenerator[DummyModel, None]:
    """In-process dummy engine for health check tests.

    Controlled via HTTP at ``{endpoint}/__control/`` endpoints.
    See ``DummyModel`` docstring for the control API.
    """
    port = get_port()
    config = DummyModelConfig(
        id="openai-mock",
        task="chat",
        max_requests=4,
        port=port,
    )
    model = DummyModel(config, lambda: None)
    await model.start()
    try:
        yield model
    finally:
        await model.terminate()


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

    model = LlamaModel(config, manager.manifest_changed.set)
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
async def gemma_real(models_cache: Path, manager: Manager) -> LlamaModel:
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
async def gemma_dummy(manager: Manager) -> DummyModel:
    """In-process dummy model for testing (no real llama-server required)."""
    config = DummyModelConfig(
        id="gemma",
        task="chat",
        max_requests=4,
        port=get_port(),
    )
    model = DummyModel(config, manager.manifest_changed.set)
    manager.add_model(model)
    return model


@pytest.fixture
async def gemma(gemma_dummy: DummyModel) -> DummyModel:
    """Default gemma fixture — uses the in-process dummy engine.

    Tests that need the real llama-server subprocess should use
    ``gemma_real`` instead.
    """
    return gemma_dummy


@pytest.fixture
async def embeddinggemma_real(
    models_cache: Path, manager: Manager
) -> LlamaModel:
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
